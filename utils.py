import torch


@torch.no_grad()
def test(model, X, y):
    model.eval()

    predicted = model.forward(X)

    _, predicted = torch.max(predicted.data, 1)

    total = y.shape[0]
    correct = (predicted == y).sum().item()

    return correct/total