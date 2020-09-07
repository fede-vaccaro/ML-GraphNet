from loader import load_graph
from sklearn.model_selection import StratifiedShuffleSplit as Split
import torch.nn.functional as F
from models import *
import utils as u
from sklearn.preprocessing import normalize
import time
import numpy as np
import data_visualization as dv
import random
import argparse
from utils import seed
# implementation of SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS - Kipf & Willing 2017
# https://arxiv.org/pdf/1609.02907.pdf

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset", type=str, default="cora",
                help="Choice one between 'cora', 'citeseer', 'facebook', 'pubmed'")
ap.add_argument("-v", "--visualize", action='store_true',
                help="For rendering embeddings (only available for 'cora' dataset)")
ap.add_argument("-dv", "--device", type=str, default='cpu',
                help="Visualize embeddings (just for 'cora' dataset)")
ap.add_argument("-k", "--kcross", action='store_true', default=False,
                help="K-cross validation on 10 folds")

args = vars(ap.parse_args())

dataset = args['dataset']
visualize = args['visualize']
device_ = args['device']
kcross = args['kcross']

if not kcross:
    torch.random.manual_seed(seed // 3)
    random.seed(seed // 3)
    np.random.seed(seed // 3)

cuda_is_available = torch.cuda.is_available()
print("Cuda: ", cuda_is_available)

if cuda_is_available and device_ == 'gpu':
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dataset_name = dataset
A, X, Y = load_graph(dataset_name)


D_inv = 1 / A.sum(axis=1).astype('float32')
P = np.dot(np.diag(D_inv), A)

A = torch.tensor(A)
P = torch.tensor(P)

feat_size = X.shape[1]

def train_dvna(A, P, verbose=False):
    n_epochs = 10000

    train_ones_indices, train, val, test = u.split_dataset(A, seed=seed)

    A_train = u.prepare_train_matrix_dvne(A, train_ones_indices)

    # sample triplets
    # nbrs is a dict nbrs[i] -> {j1,j2,...,}
    nbrs = {}
    not_nbrs = {}
    all_nodes = {i for i in range(A.shape[0])}

    for ij in zip(train_ones_indices[0], train_ones_indices[1]):
        i, j = int(ij[0]), int(ij[1])
        if i in nbrs.keys():
            nbrs[i] = nbrs[i].union({j})
        else:
            nbrs[i] = {j}

        if j in nbrs.keys():
            nbrs[j] = nbrs[j].union({i})
        else:
            nbrs[j] = {i}

    for i in nbrs.keys():
        nbrs_set = nbrs[i]
        not_nbrs[i] = all_nodes - nbrs_set

    model = DVNE(n_features=A.shape[0])
    model.to(device)

    opt = torch.optim.RMSprop(lr=0.001, params=model.parameters())

    nonzero_ratio = A.sum() / (A.shape[0] ** 2)
    zero_ratio = 1 - nonzero_ratio

    A = A.to(device)
    P = P.to(device)
    A_train = A_train.to(device)

    model.n_samples = len(train)
    criterion = l.dvne_loss

    for e in range(n_epochs):
        t0 = time.time()

        triplets = u.sample_triplets(nbrs, not_nbrs, 300)
        i, j, k = triplets

        model.train()
        opt.zero_grad()

        out_i, mi, stdi = model.forward(A_train[i, :])
        out_j, mj, stdj = model.forward(A_train[j, :])
        out_k, mk, stdk = model.forward(A_train[k, :])

        gt_i = P[i, :]
        gt_j = P[j, :]
        gt_k = P[k, :]

        out_reconstruction = torch.cat([out_i, out_j, out_k], dim=0).view(-1)
        gt = torch.cat([gt_i, gt_j, gt_k], dim=0).view(-1).to(device)
        a_gt = torch.cat([A[i, :], A[j, :], A[j, :]], dim=0).view(-1).to(device)

        x = torch.ones(gt.shape[0]).to(device)
        weights = torch.where(gt == 0.0, x * float(nonzero_ratio), x * float(zero_ratio)).to(device)
        loss_norm = weights.sum() / len(train)

        loss_weight = 0.6
        # loss = criterion(out_reconstruction, gt_reconsturction, weight=weights) * loss_weight/ loss_norm
        l2 = criterion(gt, out_reconstruction, torch.cat([stdi, stdj, stdk], dim=0)) * loss_weight

        # loss = 0.0
        w_ij = model.wasserstein((mi, stdi), (mj, stdj))
        w_ik = model.wasserstein((mi, stdi), (mk, stdk))
        l1 = l.energy_loss(w_ij, w_ik)

        loss = l1 + loss_weight*l2

        loss.backward()
        opt.step()

        t1 = time.time()
        if verbose:
            if (e + 1) % 10 == 0:
                if len(val) > 0:
                    # with torch.no_grad():
                    #     val_loss = float(
                    #         criterion((model.forward(A_train)[0]).reshape(-1)[val], A.reshape(-1)[val].data))
                    val_loss = 0
                    val_auc = u.test_auc_dvna(model, A_train, A, val)
                else:
                    val_auc = np.nan
                    val_loss = np.nan
                print(
                    "Iteration: {0}; train loss: {1:.4f}; val loss: {2:.4f}; val auc: {3:.4f}; time: {4:.4f}".format(
                        e + 1, loss,
                        val_loss,
                        val_auc,
                        t1 - t0))

    test_auc = u.test_auc_dvna(model, A_train, A, idx=test, test=True)
    # print("Test auc: ", test_auc)

    if dataset_name == 'cora' and visualize:
        with torch.no_grad():
            encodings, mean, std = model.encode(P.to(device))
            embeddings = torch.cat([mean, std], dim=1)

            dv.reduct_and_visualize(encodings.cpu().numpy(), Y.argmax(axis=1))

            train, val_test = next(Split(train_size=140, random_state=seed).split(embeddings, Y))

            embeddings = embeddings.cpu()
            x_train, y_train = embeddings[train], Y[train]
            x_test, y_test = embeddings[val_test], Y[val_test]

            svm = SVC(C=10.0)

            svm.fit(x_train, y_train.argmax(axis=1))
            y_predicted = svm.predict(x_test)
            print("SVM Accuracy: ", accuracy_score(y_predicted, y_test.argmax(axis=1)))
    return test_auc


if kcross:
    aucs = []
    for i in range(10):
        print("Training model {}/10".format(i + 1))
        auc = train_dvna(A, P, False)
        print("Auc {}: ".format(i+1), auc)
        aucs += [auc]
    print("avg AUC over 10 folds: ", np.asarray(aucs).mean())
    print("AUC std over 10 folds: ", np.asarray(aucs).std())
else:
    auc = train_dvna(A, P, True)
    print("Test auc: ", auc)
