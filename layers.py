import torch
import torch.nn as nn
import math
import utils as u
from torch.nn import init




class KipfAndWillingConv(nn.Module):
    def __init__(self, n_features, n_filters):
        super().__init__()

        self.filters = torch.nn.Parameter(torch.Tensor(n_features, n_filters), requires_grad=True)
        self.reset_parameters()

    def forward(self, x, transform):
        XF = torch.mm(x, self.filters)
        out = torch.sparse.mm(transform, XF)
        return out

    def reset_parameters(self):
        init.xavier_normal_(self.filters)

    @staticmethod
    def compute_transform(A):
        D_diag = torch.sum(A, dim=1).pow(-0.5)
        D_diag[D_diag == float("Inf")] = 0

        D = torch.diag(D_diag)

        out = D.mm(A).mm(D)
        return out


def decode(x: torch.Tensor):
    d = x.mm(x.t())
    # d = torch.sigmoid(d)
    return d
