import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphMLP(nn.Module):
    def __init__(self, input_feature_size, n_classes, hidden_size=128):
        super().__init__()
        self.hidden_size = 64

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
        self.transform = self.compute_transform(A)

        self.filters = torch.nn.Parameter(torch.Tensor(n_features, n_filters), requires_grad=True)

    def forward(self, x):
        x = self.transform.mm(x).mm(self.filters)
        return x

    @staticmethod
    def compute_transform(A):
        A_tilde = A + torch.eye(A.shape[0])
        D_tilde_diag = torch.diag(torch.sum(dim=0)).inverse().pow(0.5)

        return A_tilde.mm(D_tilde_diag)


class GCNKipf(nn.Module):
    def __init__(self, A, n_filters, n_features, n_classes):
        self.conv1 = KipfAndWillingConv(A, n_filters, n_features)
        self.conv2 = KipfAndWillingConv(A, n_classes, n_filters)

    def forward(self, x, A=None):
        if A is not None:
            transform = KipfAndWillingConv.compute_transform(A)
            self.conv1.transform = transform
            self.conv2.transform = transform

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        out = F.softmax(x, dim=1)

        return out
