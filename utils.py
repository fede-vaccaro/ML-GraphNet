import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import random


# plt.switch_backend('TkAgg') #TkAgg (instead Qt4Agg)


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
def test_auc(model, X, A, idx, test=False):
    model.eval()

    # fn = torch.sigmoid
    fn = lambda x : x

    a_cap = fn(model.forward(X))
    predicted = a_cap.view(-1)[idx].cpu().numpy()

    # if test:
    #     a_cap = a_cap.cpu().numpy()
    #     a_cap[a_cap <= 0.75] = 0
    #     a_cap[a_cap > 0.75] = 1
    #
    #     plt.matshow(a_cap)
    #     plt.show()

    groundtruth = A.view(-1)[idx].cpu().numpy()

    auc_score = roc_auc_score(y_score=predicted, y_true=groundtruth)
    if test:
        fpr, tpr, thresholds = roc_curve(y_score=predicted, y_true=groundtruth)
        # plot_roc(fpr, tpr)
        # print("AUC: ", auc(fpr, tpr))

    return auc_score


def plot_roc(fpr, tpr):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def dense_to_sparse(dense_matrix):
    indices = torch.nonzero(dense_matrix).t()
    values = dense_matrix[indices[0], indices[1]]  # modify this based on dimensionality
    out = torch.sparse.FloatTensor(indices, values, dense_matrix.size())
    return out


def indices_from_2d_to_1d(indices_list, side):
    # transforms a list of (i,j) indices (indices_list) from a matrix with shape (side,side)
    # to a single element index k.
    # equivalent to k = i + side*j
    all_indices = np.arange(side ** 2).reshape((side, side))
    indices = all_indices[indices_list].reshape(-1)
    return indices


def split_dataset(A, seed):
    A_integer = A.cpu().numpy().astype('int32')

    edges_indices = np.triu_indices(A.shape[0])

    # deletes diagonal elements from datasets
    mask = edges_indices[0] != edges_indices[1]
    edges_indices = edges_indices[0][mask], edges_indices[1][mask]

    values = A_integer[edges_indices]

    arg_ones_values = np.argwhere(values > 0)
    arg_zero_values = np.argwhere(values == 0)

    indices_ones = np.random.RandomState(seed=seed).permutation(len(arg_ones_values))
    indices_zeros = np.random.RandomState(seed=seed).permutation(len(arg_zero_values))

    # we want the test set is made by 10% of the total 1s
    n_ones_test = (0.10 * len(arg_ones_values)).__int__()
    n_zeros_test = (0.10 * len(arg_zero_values)).__int__()

    n_ones_val = (0.15 * len(arg_ones_values)).__int__()
    n_zeros_val = (0.15 * len(arg_zero_values)).__int__()

    n_ones_train = (len(arg_ones_values)).__int__()
    n_zeros_train = (len(arg_zero_values)).__int__()

    indices_test_ones = indices_ones[:n_ones_test]
    indices_test_zeros = indices_zeros[:n_zeros_test]

    indices_val_ones = indices_ones[n_ones_test:n_ones_val]
    indices_val_zeros = indices_zeros[n_zeros_test:n_zeros_val]

    indices_train_ones = indices_ones[n_ones_val:n_ones_train]
    indices_train_zeros = indices_zeros[n_zeros_val:n_zeros_train]

    # defines indices of edges that will go into the test set
    # these are not the indices of the test set!
    test_set_edge_indices = np.vstack((arg_ones_values[indices_test_ones], arg_zero_values[indices_test_zeros]))
    val_set_edge_indices = np.vstack((arg_ones_values[indices_val_ones], arg_zero_values[indices_val_zeros]))
    train_set_edge_indices = np.vstack((arg_ones_values[indices_train_ones], arg_zero_values[indices_train_zeros]))

    train = edges_indices[0][train_set_edge_indices], edges_indices[1][train_set_edge_indices]
    train_ones = edges_indices[0][arg_ones_values[indices_train_ones]], edges_indices[1][
        arg_ones_values[indices_train_ones]]

    val = edges_indices[0][val_set_edge_indices], edges_indices[1][val_set_edge_indices]
    val_ones = edges_indices[0][arg_ones_values[indices_val_ones]], edges_indices[1][arg_ones_values[indices_val_ones]]

    test = edges_indices[0][test_set_edge_indices], edges_indices[1][test_set_edge_indices]
    test_ones = edges_indices[0][arg_ones_values[indices_test_ones]], edges_indices[1][
        arg_ones_values[indices_test_ones]]

    # x = torch.tensor(A.data).numpy()
    # x *= 0
    #
    # x[test_ones] = 1
    # plt.matshow(x)
    # plt.show()
    #
    # x[train_ones] = 1
    # plt.matshow(x)
    # plt.show()
    #
    # x[val_ones] = 1
    # plt.matshow(x)
    # plt.show()

    return train_ones, indices_from_2d_to_1d(train, A.shape[0]), indices_from_2d_to_1d(val, A.shape[
        0]), indices_from_2d_to_1d(test, A.shape[0])
