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

# implementation of SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS - Kipf & Willing 2017
# https://arxiv.org/pdf/1609.02907.pdf
# and "Variational Graph Auto-Encoders" - Kipf & Willing 2016
# https://arxiv.org/pdf/1611.07308.pdf

seed = (2 ** 31 - 53423)
torch.random.manual_seed(seed // 3)
random.seed(seed // 3)
np.random.seed(seed // 3)

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

args = vars(ap.parse_args())

dataset = args['dataset']
method = args['method']
visualize = args['visualize']
device_ = args['device']
use_features = args['features']


dataset_name = dataset
A, X, Y = load_graph(dataset_name)

A = torch.tensor(A.astype('float32'))

if use_features:
    assert dataset_name == 'cora', "Node features available just for 'cora' dataset"
    print("Using features")
    X = torch.tensor(X.astype('float32'))
else:
    X = None
    # X = torch.eye(A.shape[0]).type(torch.float32)


n_epochs = 400
n_splits = 10

feat_size = X.shape[1] if X else A.shape[0]

cuda_is_available = torch.cuda.is_available()
print("Cuda: ", cuda_is_available)

if cuda_is_available and device_ == 'gpu':
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# criterion = F.binary_cross_entropy
criterion = F.binary_cross_entropy_with_logits

val_history = []
val_eps_early_stop = 1e-6

not_improving_counter = 0
not_improving_max_step = n_epochs

print("Not improving epsilon: ", val_eps_early_stop)

# A += torch.eye(A.shape[0])
train_, train, val, test = u.split_dataset(A, seed=seed)

A_train = np.zeros(A.shape).astype('float32')
A_train[train_] = 1
A_train += A_train.T
np.fill_diagonal(A_train, 1)
A_train = torch.from_numpy(A_train)
A_train = A_train.type(torch.float32)

# plt.matshow(A_model.numpy())
# plt.show()

if method == 'gae':
    model = GCNAutoencoder(n_features=feat_size, hidden_dim=32, code_dim=16, A=A_train)
elif method == 'vgae':
    model = GcnVAE(n_features=feat_size, n_samples=A.shape[0], hidden_dim=32, code_dim=16, A=A_train)
else:
    raise ValueError("Method {} not available!".format(method))

print("Using method: ", method)

model.to(device)
opt = torch.optim.Adam(lr=0.01, params=model.parameters())

lambda_reg = 5e-4

nonzero_ratio = A.sum() / (A.shape[0] ** 2)
zero_ratio = 1 - nonzero_ratio

if X:
    X = X.to(device)
A = A.to(device)

model.n_samples = len(train)

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
    if (e + 1) % 10 == 0:
        val_auc = u.test_auc_gae(model, X, A, val)
        with torch.no_grad():
            val_loss = float(criterion(forward.reshape(-1)[val], A.reshape(-1)[val].data))

        print(
            "Epoch: {0}; train loss: {1:.4f}; val loss: {2:.4f}; val auc: {3:.4f}; time: {4:.4f}".format(e + 1, loss,
                                                                                                         val_loss,
                                                                                                         val_auc,
                                                                                                         t1 - t0))

test_auc = u.test_auc_gae(model, X, A, test, test=True)
print("Test auc: ", test_auc)

if dataset_name == 'cora' and visualize:
    with torch.no_grad():
        encoded, _, _ = model.encode(X)
        dv.reduct_and_visualize(encoded.cpu().numpy(), Y.argmax(axis=1))

    train, val_test = next(Split(train_size=140, random_state=seed).split(encoded, Y))

    # encoded = encoded.cpu()
    # x_train, y_train = encoded[train], Y[train]
    # x_test, y_test = encoded[val_test], Y[val_test]
    #
    # svm = SVC(C=10.0)
    #
    # svm.fit(x_train, y_train.argmax(axis=1))
    # y_predicted = svm.predict(x_test)
    # print("Accuracy: ", accuracy_score(y_predicted, y_test.argmax(axis=1)))
