import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def test_acc(model, X, y, idx):
    model.eval()

    predicted = model.forward(X)[idx]
    _, predicted = torch.max(predicted.data, 1)

    y = y.argmax(dim=1)[idx]

    total = y.shape[0]
    correct = (predicted == y).sum().item()

    return correct / total


@torch.no_grad()
def test_auc(model, X, A, idx):
    model.eval()
    predicted = model.forward(X).view(-1)[idx].numpy()

    groundtruth = torch.clamp(A, max=2.0).view(-1)[idx].numpy().astype('int32')

    auc = roc_auc_score(y_score=predicted, y_true=groundtruth)
    return auc


def dense_to_sparse(dense_matrix):
    indices = torch.nonzero(dense_matrix).t()
    values = dense_matrix[indices[0], indices[1]]  # modify this based on dimensionality
    out = torch.sparse.FloatTensor(indices, values, dense_matrix.size())
    return out
