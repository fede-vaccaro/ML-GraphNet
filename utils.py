import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import random
from torch import nn as nn
seed = (2 ** 31 - 53423)
np.random.seed(seed // 3)


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
def test_auc_dvna(model, X, A, idx, test=False):
    model.eval()

    # fn = torch.sigmoid

    _, means, std = model.encode(X)

    dist_m = torch.cdist(means, means).pow(2)
    dist_s = torch.cdist(std, std).pow(2)

    a_cap = dist_m + dist_s
    a_cap = (a_cap.max() - a_cap) / a_cap.max()

    predicted = a_cap.reshape(-1)[idx].cpu().numpy()

    # if test:
    #     a_cap = a_cap.cpu().numpy()
    #     a_cap[a_cap <= 0.75] = 0
    #     a_cap[a_cap > 0.75] = 1
    #
    #     plt.matshow(a_cap)
    #     plt.show()

    groundtruth = A.reshape(-1)[idx].cpu().numpy()

    auc_score = roc_auc_score(y_score=predicted, y_true=groundtruth)
    if test:
        fpr, tpr, thresholds = roc_curve(y_score=predicted, y_true=groundtruth)
        # plot_roc(fpr, tpr)
        # print("AUC: ", auc(fpr, tpr))

    return auc_score


@torch.no_grad()
def test_auc_gae(model, X, A, idx, test=False):
    model.eval()

    fn = torch.sigmoid
    a_cap, _, _ = model.encode(X)

    a_cap = a_cap / a_cap.norm(p=2, dim=0)
    a_cap = a_cap @ a_cap.t()
    a_cap = a_cap.cpu()

    predicted = a_cap.view(-1)[idx].numpy()

    # if test:
    #     a_cap = a_cap.cpu().numpy()
    #     a_cap[a_cap <= 0.75] = 0
    #     a_cap[a_cap > 0.75] = 1
    #
    #     plt.matshow(a_cap)
    #     plt.show()

    groundtruth = A.reshape(-1)[idx].cpu().numpy()

    auc_score = roc_auc_score(y_score=predicted, y_true=groundtruth)
    if test:
        fpr, tpr, thresholds = roc_curve(y_score=predicted, y_true=groundtruth)
        # plot_roc(fpr, tpr)
        # print("AUC: ", auc(fpr, tpr))

    return auc_score


def sample_triplets(nbrs, not_nbrs, n_triplets):
    assert n_triplets <= len(nbrs.keys())
    assert nbrs.keys() == not_nbrs.keys()

    nodes_ids = np.random.permutation(len(nbrs.keys()))[:n_triplets]

    nodes = np.array(list(nbrs.keys()))
    nodes = nodes[nodes_ids]
    positives = []
    negatives = []

    for node in nodes:
        positives += [random.sample(nbrs[node], 1)[0]]
        negatives += [random.sample(not_nbrs[node], 1)[0]]

    return nodes, positives, negatives


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

def prepare_train_matrix_dvne(A, indices_ones):
    A_train = np.zeros(A.shape).astype('float32')
    A_train[indices_ones] = 1
    A_train += A_train.T
    A_train_d_inv = 1 / A_train.sum(axis=1)
    A_train_d_inv[A_train_d_inv == np.inf] = 0
    A_train_d_inv = np.diag(A_train_d_inv)
    A_train = A_train.dot(A_train_d_inv)
    A_train = torch.from_numpy(A_train)
    A_train = A_train.type(torch.float32)
    return A_train


def prepare_train_matrix_gae(A, indices_one):
    A_train = np.zeros(A.shape).astype('float32')
    A_train[indices_one] = 1
    A_train += A_train.T
    np.fill_diagonal(A_train, 1)
    A_train = torch.from_numpy(A_train)
    A_train = A_train.type(torch.float32)
    return A_train

def split_dataset(A, seed):
    A_integer = A.cpu().numpy().astype('int32')

    edges_indices = np.triu_indices(A.shape[0])

    # deletes diagonal elements from datasets
    mask = edges_indices[0] != edges_indices[1]
    edges_indices = edges_indices[0][mask], edges_indices[1][mask]

    values = A_integer[edges_indices]

    # argwhere is not deterministic as it runs more likely on multiple threads
    arg_ones_values = np.argwhere(values > 0)
    arg_zero_values = np.argwhere(values == 0)

    indices_ones = np.random.permutation(len(arg_ones_values))
    indices_zeros = np.random.permutation(len(arg_zero_values))

    test_split = 0.10
    val_split = 0.15

    # train split = 1.0 - val_split

    # we want the test set is made by 10% of the total 1s
    n_ones_test = (test_split * len(arg_ones_values)).__int__()
    n_zeros_test = (test_split * len(arg_zero_values)).__int__()

    n_ones_val = (val_split * len(arg_ones_values)).__int__()
    n_zeros_val = (val_split * len(arg_zero_values)).__int__()

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
    train_ones_indices = edges_indices[0][arg_ones_values[indices_train_ones]], edges_indices[1][
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
    # x[train_ones_indices] = 1
    # plt.matshow(x)
    # plt.show()
    #
    # x[val_ones] = 1
    # plt.matshow(x)
    # plt.show()

    return train_ones_indices, indices_from_2d_to_1d(train, A.shape[0]), indices_from_2d_to_1d(val, A.shape[
        0]), indices_from_2d_to_1d(test, A.shape[0])
