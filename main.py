from loader import load_graph
from sklearn.model_selection import StratifiedShuffleSplit as Split
import torch
from models import *
import utils as u

A, X, Y = load_graph('cora')

A = torch.tensor(A.astype('float32'))
X = torch.tensor(X.astype('float32'))
Y = torch.tensor(Y.astype('float32'))

n_epochs = 300

feat_size = X.shape[1]
n_classes = Y.shape[1]

model = GraphMLP(input_feature_size=feat_size, n_classes=n_classes)
opt = torch.optim.Adam(lr=0.01, params=model.parameters())
criterion = torch.nn.CrossEntropyLoss()

val_history = []
val_eps_early_stop = 0.0

not_improving_counter = 0
not_improving_max_step = 10

print("Not improving epsilon: ", val_eps_early_stop)

train_val, test = next(Split(train_size=280).split(X, Y))

train = train_val[:140]
val = train_val[140:280]


for e in range(n_epochs):
    model.train()

    opt.zero_grad()

    out = model.forward(X[train])
    loss = criterion(out, Y[train].sum(dim=1))

    loss.backward()
    opt.step()

    print("Loss at epoch {}: {}".format(e, loss))

    val_acc = u.test(model, X[val], Y[val])
    print("Val accuracy: ", val_acc)

    with torch.no_grad():
        val_loss = float(criterion(model.forward(X[val]), Y[val]).data)
        print("Val loss: ", val_loss)

    if e > 0:
        if val_loss < val_history[-1] - val_eps_early_stop:
            not_improving_counter = 0
        elif not_improving_counter > not_improving_max_step:
            print("Breaking at epoch: ", e)
            break
        else:
            not_improving_counter += 1

    val_history += [val_loss]
    print()

print("Test accuracy: ", u.test(model, X[test], Y[test]))
