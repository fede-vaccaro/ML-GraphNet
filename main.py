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

plt.matshow(A)
# plt.show()

seed = (2 ** 31)
torch.random.manual_seed(seed // 3)

# X = normalize(X.astype('float32'), 'l1')

A = torch.tensor(A.astype('float32'))
# X = torch.eye(A.shape[0]).type(torch.float32)
X = torch.tensor(X.astype('float32'))

n_epochs = 400

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

train_, train, val, test = u.split_dataset(A, seed=seed)

A_model = np.zeros(A.shape).astype('float32')
A_model[train_] = 1
A_model += A_model.T

A_model = torch.from_numpy(A_model)
A_model = A_model.type(torch.float32)

# plt.matshow(A_model.numpy())
# plt.show()

# model = GCNAutoencoder(n_features=feat_size, hidden_dim=32, code_dim=16, A=A_model)
model = GcnVAE(n_features=feat_size, n_samples=A.shape[0], hidden_dim=32, code_dim=16, A=A_model)
model.to(device)

opt = torch.optim.Adam(lr=0.01, params=model.parameters())

lambda_reg = 5e-4

nonzero_ratio = A.sum() / (A.shape[0] ** 2)
zero_ratio = 1 - nonzero_ratio

X = X.to(device)
A = A.to(device)

model.n_samples = len(train)

for e in range(n_epochs):
    t0 = time.time()
    model.train()
    opt.zero_grad()

    forward = model.forward(X)
    out = forward.view(-1)[train]
    ground_truth_links = A.view(-1)[train]

    x = torch.ones(ground_truth_links.shape[0]).to(device)
    weights = torch.where(ground_truth_links < 0.5, x * nonzero_ratio, x * zero_ratio).to(device)
    loss = criterion(out, ground_truth_links, weight=weights) + model.kl_divergence()

    loss.backward()
    opt.step()

    val_auc = u.test_auc(model, X, A, val)

    with torch.no_grad():
        val_loss = float(criterion(forward.view(-1)[val], A.view(-1)[val].data))

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

print("Test auc: ", u.test_auc(model, X, A, test, test=True))
