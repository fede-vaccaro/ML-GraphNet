import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as u

class GraphMLP(nn.Module):
    def __init__(self, input_feature_size, n_classes, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size

        self.hidden = nn.Linear(input_feature_size, hidden_size)
        self.output = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)

        x = F.dropout(x, p=0.5, training=self.training)

        x = self.output(x)
        y = F.softmax(x, dim=1)

        return y


class KipfAndWillingConv(nn.Module):
    def __init__(self, A, n_filters, n_features):
        super().__init__()
        transform = self.compute_transform(A + torch.eye(A.shape[0]))
        self.transform = u.dense_to_sparse(transform)

        self.filters = torch.nn.Parameter(torch.Tensor(n_features, n_filters), requires_grad=True)
        stdv = 1. / math.sqrt(self.filters.size(1))
        self.filters.data.uniform_(-stdv, stdv)

    def forward(self, x):
        XF = torch.mm(x, self.filters)
        out = torch.sparse.mm(self.transform, XF)
        return out

    @staticmethod
    def compute_transform(A):
        D_diag = torch.sum(A, dim=1).pow(-1)
        D = torch.diag(D_diag)

        # same as D_diag.pow(-0.5) and then
        # D * A * D since both D and A are symmetric

        out = D.mm(A)
        return out


class GCNKipf(nn.Module):
    def __init__(self, A, n_filters, n_features, n_classes):
        super().__init__()
        self.conv1 = KipfAndWillingConv(A, n_filters, n_features)
        self.out = KipfAndWillingConv(A, n_classes, n_filters)

    def forward(self, x, A=None):
        if A is not None:
            transform = KipfAndWillingConv.compute_transform(A)
            self.conv1.transform = transform
            self.out.transform = transform

        x = self.conv1(x)
        x = F.relu(x)

        x = F.dropout(x, p=0.5, training=self.training)

        x = self.out(x)
        out = F.softmax(x, dim=1)

        return out
