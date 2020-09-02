from loader import load_graph
from sklearn.model_selection import StratifiedShuffleSplit as Split
import torch.nn.functional as F
from models import *
import utils as u
from sklearn.preprocessing import normalize
import time
import numpy as np
import matplotlib.pyplot as plt

# implementation of SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS - Kipf & Willing 2017
# https://arxiv.org/pdf/1609.02907.pdf
# and "Variational Graph Auto-Encoders" - Kipf & Willing 2016
# https://arxiv.org/pdf/1611.07308.pdf

print(torch.cuda.is_available())

A, X, Y = load_graph('cora')

# plt.matshow(A)
# plt.show()

seed = (2 ** 31)
torch.random.manual_seed(seed // 3)
# X = normalize(X.astype('float32'), 'l1')

A = torch.tensor(A.astype('float32'))
# X = torch.eye(A.shape[0]).type(torch.float32)
X = torch.tensor(X.astype('float32'))


D_inv = A.sum(dim=1).pow(-1)
P = torch.diag(D_inv).matmul(A)

n_epochs = 4000
n_splits = 10

feat_size = X.shape[1]
n_classes = Y.shape[1]

device = torch.device("cuda")

# criterion = F.binary_cross_entropy
criterion = F.binary_cross_entropy_with_logits
import torch.nn as nn

class DVNELoss(nn.Module):
    def __init__(self):
        super(DVNELoss, self).__init__()

    def forward(self, gt, pred, weights=None):
        expected = ((gt - pred)) ** 2
        expected = expected[torch.where(expected > 0)]
        #if weights is not None:
        #    expected *= weights
        expected = expected.mean()
        return expected

criterion = DVNELoss()

val_history = []
val_eps_early_stop = 1e-6

not_improving_counter = 0
not_improving_max_step = n_epochs

print("Not improving epsilon: ", val_eps_early_stop)

# A += torch.eye(A.shape[0])
train_ones_indices, train, val, test = u.split_dataset(A, seed=seed)

A_model = np.zeros(A.shape).astype('float32')
A_model[train_ones_indices] = 1
A_model += A_model.T
# np.fill_diagonal(A_model, 1)

#########################################
## transform A_model into transition matrix

A_model_d_inv = 1/A_model.sum(axis=1)
A_model_d_inv[A_model_d_inv == np.inf] = 0
A_model_d_inv = np.diag(A_model_d_inv)
A_model = A_model.dot(A_model_d_inv)

########################################

A_model = torch.from_numpy(A_model)
A_model = A_model.type(torch.float32)
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

# plt.matshow(A_model.numpy())
# plt.show()

# model = GCNAutoencoder(n_features=feat_size, hidden_dim=32, code_dim=16, A=A_model)
model = DVNE(n_samples=A.shape[0], n_features=A.shape[0])
model.to(device)

opt = torch.optim.Adam(lr=0.001, params=model.parameters())

lambda_reg = 5e-4

nonzero_ratio = A.sum() / (A.shape[0] ** 2)
zero_ratio = 1 - nonzero_ratio

X = X.to(device)
A = A.to(device)
A_model = A_model.to(device)

model.n_samples = len(train)


def energy_loss(w_ij, w_ik):
    energy = w_ij.pow(2.0) + torch.exp(-w_ik)
    return energy.mean()


for e in range(n_epochs):
    t0 = time.time()

    triplets = u.sample_triplets(nbrs, not_nbrs, 200)
    i, j, k = triplets

    model.train()
    opt.zero_grad()

    out_i, mi, stdi = model.forward(A_model[i, :])
    out_j, mj, stdj = model.forward(A_model[j, :])
    out_k, mk, stdk = model.forward(A_model[k, :])

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
    loss = criterion(gt, out_reconstruction, a_gt) * loss_weight

    # loss = 0.0
    if isinstance(model, DVNE):
        w_ij = model.wasserstein((mi, stdi), (mj, stdj))
        w_ik = model.wasserstein((mi, stdi), (mk, stdk))
        t = energy_loss(w_ij, w_ik)
        loss += t

    if isinstance(model, GcnVAE):
        loss += model.kl_divergence()

    loss.backward()
    opt.step()

    t1 = time.time()
    if (e + 1) % 10 == 0:
        if len(val) > 0:
            with torch.no_grad():
                val_loss = float(criterion((model.forward(A_model)[0]).reshape(-1)[val], A.reshape(-1)[val].data))
            val_auc = u.test_auc_dvna(model, A_model, A, val)
        else:
            val_auc = np.nan
            val_loss = np.nan
        print(
            "Epoch: {0}; train loss: {1:.4f}; val loss: {2:.4f}; val auc: {3:.4f}; time: {4:.4f}".format(e + 1, loss,
                                                                                                         val_loss,
                                                                                                         val_auc,
                                                                                                         t1 - t0))

test_auc = u.test_auc_dvna(model, A_model, A, idx=test, test=True)
print("Test auc {}: ", test_auc)
