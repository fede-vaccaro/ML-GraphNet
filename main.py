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

n_epochs = 1000
n_splits = 10

feat_size = X.shape[1]
n_classes = Y.shape[1]

device = torch.device("cuda")

# criterion = F.binary_cross_entropy
criterion = F.binary_cross_entropy_with_logits

val_history = []
val_eps_early_stop = 1e-6

not_improving_counter = 0
not_improving_max_step = n_epochs

print("Not improving epsilon: ", val_eps_early_stop)

# A += torch.eye(A.shape[0])
train_, train, val, test = u.split_dataset(A, seed=seed)

A_model = np.zeros(A.shape).astype('float32')
A_model[train_] = 1
A_model += A_model.T
# np.fill_diagonal(A_model, 1)
A_model = torch.from_numpy(A_model)
A_model = A_model.type(torch.float32)

# sample triplets
# nbrs is a dict nbrs[i] -> {j1,j2,...,}
nbrs = {}
not_nbrs = {}

for ij in zip(train_[0], train_[1]):
    i, j = int(ij[0]), int(ij[1])
    if i in nbrs.keys():
        nbrs[i] = nbrs[i].union({j})
    else:
        nbrs[i] = {j}

for i in nbrs.keys():
    nbrs_set = nbrs[i]
    all_nodes = {i for i in range(A.shape[0])}
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
    return w_ij.pow(2.0) + torch.exp(-w_ij)


for e in range(n_epochs*10):
    t0 = time.time()

    triplets = u.sample_triplets(nbrs, not_nbrs, 200)
    ids, i, j, k = triplets

    model.train()
    opt.zero_grad()

    out_i, mi, stdi = model.forward(A_model[i, :])
    out_j, mj, stdj = model.forward(A_model[j, :])
    out_k, mk, stdk = model.forward(A_model[k, :])

    gt_i = A[i, :]
    gt_j = A[j, :]
    gt_k = A[k, :]

    out_reconstruction = torch.cat([out_i, out_j, out_k], dim=0).view(-1)
    gt_reconsturction = torch.cat([gt_i, gt_j, gt_k], dim=0).view(-1)

    x = torch.ones(gt_reconsturction.shape[0]).to(device)
    weights = torch.where(gt_reconsturction < 0.5, x * nonzero_ratio, x * zero_ratio).to(device)
    loss_norm = weights.sum() / len(train)

    loss = criterion(out_reconstruction, gt_reconsturction, weight=weights)*0.6
    if isinstance(model, DVNE):
        w_ij = model.wasserstein((mi, stdi), (mj, stdj))
        w_ik = model.wasserstein((mi, stdi), (mk, stdk))
        loss += energy_loss(w_ij, w_ik)

    if isinstance(model, GcnVAE):
        loss += model.kl_divergence()

    loss.backward()
    opt.step()

    val_auc = u.test_auc(model, A_model, A, val)

    with torch.no_grad():
        val_loss = float(criterion((model.forward(A_model)[0]).view(-1)[val], A.view(-1)[val].data))

    if e > 0:
        if val_loss < val_history[-1] - val_eps_early_stop:
            not_improving_counter = 0
        elif not_improving_counter > not_improving_max_step:
            print("Breaking at epoch: ", e)
            break
        else:
            not_improving_counter += 1

    val_history += [val_loss]
    t1 = time.time()
    if (e + 1) % 10 == 0:
        print(
            "Epoch: {0}; train loss: {1:.4f}; val loss: {2:.4f}; val auc: {3:.4f}; time: {4:.4f}".format(e + 1, loss,
                                                                                                         val_loss,
                                                                                                         val_auc,
                                                                                                         t1 - t0))

test_auc = u.test_auc(model, A_model, A, test, test=True)
print("Test auc {}: ", test_auc)
