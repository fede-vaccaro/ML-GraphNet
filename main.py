from loader import load_graph
from sklearn.model_selection import StratifiedShuffleSplit as Split
import torch
from models import *
import utils as u
from sklearn.preprocessing import normalize
import time

# implementation of SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS - Kipf & Willing 2017
# https://arxiv.org/pdf/1609.02907.pdf

A, X, Y = load_graph('cora')

seed = 2**31 - 1

A = torch.tensor(A.astype('float32'))
X = torch.tensor(X.astype('float32'))
Y = torch.tensor(Y.astype('float32'))

X = KipfAndWillingConv.compute_transform(X)

n_epochs = 300

feat_size = X.shape[1]
n_classes = Y.shape[1]

# model = GraphMLP(input_feature_size=feat_size, n_classes=n_classes)
model = GCNKipf(n_features=feat_size, n_classes=n_classes, A=A, n_filters=16)

# model.conv1.transform = A
# model.out.transform = A

opt = torch.optim.Adam(lr=0.01, params=model.parameters())
criterion = torch.nn.CrossEntropyLoss()

val_history = []
val_eps_early_stop = 1e-6

not_improving_counter = 0
not_improving_max_step = 10

print("Not improving epsilon: ", val_eps_early_stop)

train, val_test = next(Split(train_size=140, random_state=seed).split(X, Y))

val = val_test[:200]
test = val_test[200:]

lambda_reg = 5e-4

# X = u.dense_to_sparse(X)

for e in range(n_epochs):
    t0 = time.time()
    model.train()
    opt.zero_grad()

    out = model.forward(X)
    train_labels = Y[train].argmax(dim=1)
    loss = criterion(out[train], train_labels)

    # add l2 reg on first layer
    loss += model.conv1.filters.norm(p=2)*lambda_reg

    loss.backward()
    opt.step()

    val_acc = u.test(model, X, Y, val)

    with torch.no_grad():
        val_loss = float(criterion(model.forward(X)[val], Y[val].argmax(dim=1)).data)

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
    print("Epoch: {0}; train loss: {1:.4f}; val loss: {2:.4f}; val acc: {3:.4f}; time: {4:.4f}".format(e, loss, val_loss, val_acc, t1 - t0))

print("Test accuracy: ", u.test(model, X, Y, test))
