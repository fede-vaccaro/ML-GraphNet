import argparse

from loader import load_graph
from sklearn.model_selection import StratifiedShuffleSplit as Split
import torch.nn.functional as F
from models import *
import utils as u
from sklearn.preprocessing import normalize
import time
import numpy as np
import matplotlib.pyplot as plt
import data_visualization as dv
import random
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from utils import seed

# implementation of SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS - Kipf & Willing 2017
# https://arxiv.org/pdf/1609.02907.pdf
# and "Variational Graph Auto-Encoders" - Kipf & Willing 2016
# https://arxiv.org/pdf/1611.07308.pdf


ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset", type=str, default="cora",
                help="Choice one between 'cora', 'citeseer', 'facebook', 'pubmed'")
ap.add_argument("-m", "--method", type=str, default='vgae',
                help="Graph AutoEncoder (gae) or Variational Graph AutoEncoder (vgae)")
ap.add_argument("-v", "--visualize", action='store_true',
                help="Visualize embeddings (only available for 'cora' dataset)")
ap.add_argument("-dv", "--device", type=str, default='cpu',
                help="Select between 'gpu' or 'cpu'. If cuda is not available, it will run on CPU by default.")
ap.add_argument("-f", "--features", action='store_true', default=False,
                help="Use node features (only available for 'cora' dataset)")
ap.add_argument("-k", "--kcross", action='store_true', default=False,
                help="K-cross validation on 10 folds")

args = vars(ap.parse_args())

dataset = args['dataset']
method = args['method']
visualize = args['visualize']
device_ = args['device']
use_features = args['features']
kcross = args['kcross']

if not kcross:
    torch.random.manual_seed(seed // 3)
    random.seed(seed // 3)
    np.random.seed(seed // 3)

dataset_name = dataset
A, X, Y = load_graph(dataset_name)

A = torch.tensor(A.astype('float32'))

if use_features:
    assert dataset_name != 'facebook', "Node features not available for 'facebook' dataset"
    print("Using features")
    X = torch.tensor(X.astype('float32'))
else:
    X = None
    # X = torch.eye(A.shape[0]).type(torch.float32)

n_epochs = 400
n_splits = 10

feat_size = X.shape[1] if X is not None else A.shape[0]

cuda_is_available = torch.cuda.is_available()

print("Cuda: ", cuda_is_available)
if cuda_is_available and device_ == 'gpu':
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

A += torch.eye(A.shape[0])

print("Using method: ", method)


def train_gae(A, X, verbose=True):
    indices_one, train, val, test = u.split_dataset(A, seed=seed)

    A_train = u.prepare_train_matrix_gae(A, indices_one)

    if method == 'gae':
        model = GCNAutoencoder(n_features=feat_size, hidden_dim=64, code_dim=32, A=A_train)
    elif method == 'vgae':
        model = GcnVAE(n_features=feat_size, n_samples=A.shape[0], hidden_dim=64, code_dim=32, A=A_train)
    elif method == 'ae':
        model = Autoencoder(n_features=feat_size)
        if not use_features:
            X = A_train

    else:
        raise ValueError("Method {} not available!".format(method))

    model.to(device)
    opt = torch.optim.Adam(lr=0.01, params=model.parameters())

    nonzero_ratio = A.sum() / (A.shape[0] ** 2)
    zero_ratio = 1 - nonzero_ratio

    if X is not None:
        X = X.to(device)
    A = A.to(device)

    model.n_samples = len(train)

    criterion = F.binary_cross_entropy_with_logits

    for e in range(n_epochs):
        t0 = time.time()
        model.train()
        opt.zero_grad()

        forward = model.forward(X)
        out = forward.view(-1)[train]
        ground_truth_links = A.reshape(-1)[train]

        x = torch.ones(ground_truth_links.shape[0]).to(device)
        weights = torch.where(ground_truth_links < 0.5, x * nonzero_ratio, x * zero_ratio).to(device)
        loss_norm = weights.sum() / len(train)

        loss = criterion(out, ground_truth_links, weight=weights) / loss_norm
        if isinstance(model, GcnVAE):
            loss += model.kl_divergence()

        loss.backward()
        opt.step()

        t1 = time.time()
        if (e + 1) % 10 == 0 and verbose:
            val_auc = u.test_auc_gae(model, X, A, val)
            with torch.no_grad():
                val_loss = float(criterion(forward.reshape(-1)[val], A.reshape(-1)[val].data))

            print(
                "Epoch: {0}; train loss: {1:.4f}; val loss: {2:.4f}; val auc: {3:.4f}; time: {4:.4f}".format(e + 1,
                                                                                                             loss,
                                                                                                             val_loss,
                                                                                                             val_auc,
                                                                                                             t1 - t0))

    test_auc = u.test_auc_gae(model, X, A, test, test=True)

    if dataset_name == 'cora' and visualize:
        with torch.no_grad():
            encoded, _, _ = model.encode(X)
            dv.reduct_and_visualize(encoded.cpu().numpy(), Y.argmax(axis=1))

        train, val_test = next(Split(train_size=140, random_state=seed).split(encoded, Y))
        encoded = encoded.cpu()
        x_train, y_train = encoded[train], Y[train]
        x_test, y_test = encoded[val_test], Y[val_test]

        svm = SVC(C=10.0)

        svm.fit(x_train, y_train.argmax(axis=1))
        y_predicted = svm.predict(x_test)
        print("Accuracy: ", accuracy_score(y_predicted, y_test.argmax(axis=1)))

    return test_auc


if kcross:
    aucs = []
    for i in range(10):
        print("Training model {}/10".format(i + 1))
        auc = train_gae(A, X, False)
        print("Auc {}: ".format(i + 1), auc)
        aucs += [auc]
    print("avg AUC over 10 folds: ", np.asarray(aucs).mean())
    print("AUC std over 10 folds: ", np.asarray(aucs).std())
else:
    auc = train_gae(A, X, True)
    print("Test auc: ", auc)
