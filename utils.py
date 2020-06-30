import torch


@torch.no_grad()
def test(model, X, y, idx):
    model.eval()

    predicted = model.forward(X)[idx]
    _, predicted = torch.max(predicted.data, 1)

    y = y.argmax(dim=1)[idx]

    total = y.shape[0]
    correct = (predicted == y).sum().item()

    return correct/total

def dense_to_sparse(dense_matrix):
    indices = torch.nonzero(dense_matrix).t()
    values = dense_matrix[indices[0], indices[1]]  # modify this based on dimensionality
    out = torch.sparse.FloatTensor(indices, values, dense_matrix.size())
    return out