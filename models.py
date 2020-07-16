import torch
import torch.nn as nn
import torch.nn.functional as F
import layers as l
import utils as u


class GCNAutoencoder(nn.Module):
    def __init__(self, A, n_features, hidden_dim=32, code_dim=16):
        super().__init__()
        self.hidden = l.KipfAndWillingConv(n_features, hidden_dim)
        self.encoder = l.KipfAndWillingConv(hidden_dim, code_dim)

        transform = l.KipfAndWillingConv.compute_transform(A)
        self.transform = u.dense_to_sparse(transform)

    def forward(self, x):
        hidden = self.hidden(x, self.transform)
        hidden = F.relu(hidden)

        hidden = F.dropout(hidden, p=0.5, training=self.training)

        encoded = self.encoder(hidden, self.transform)
        prediction = l.decode(encoded)
        prediction = F.dropout(prediction, p=0.5, training=self.training)

        return prediction

    def reg(self, lambda_reg=1e-5):
        return self.encoder.filters.norm(p=2) * lambda_reg

    def to(self, device):
        super().to(device)
        self.transform = self.transform.to(device)


class GcnVAE(nn.Module):
    def __init__(self, A, n_features, n_samples, hidden_dim=32, code_dim=16):
        super().__init__()
        self.hidden = l.KipfAndWillingConv(n_features, hidden_dim)

        self.means_encoder = l.KipfAndWillingConv(hidden_dim, code_dim)
        self.log_std_encoder = l.KipfAndWillingConv(hidden_dim, code_dim)

        transform = l.KipfAndWillingConv.compute_transform(A)

        self.transform = u.dense_to_sparse(transform)
        self.ones = torch.ones(n_samples, code_dim)
        self.n_samples = n_samples

    def to(self, device):
        self.transform = self.transform.to(device)
        self.ones = self.ones.to(device)
        super().to(device)

    def kl_divergence(self):
        kl = 2.0 * self.log_std - self.means.pow(2.0) - (2.0 * self.log_std).exp() + 1
        kl = kl.sum(dim=1).mean() * 0.5 / self.n_samples

        return -kl

    def forward(self, x):
        hidden = self.hidden(x, self.transform)
        hidden = F.relu(hidden)

        hidden = F.dropout(hidden, p=0.5, training=self.training)

        means = self.means_encoder(hidden, self.transform)
        log_std = 0.5 * self.log_std_encoder(hidden, self.transform)
        std = torch.exp(log_std)

        # reparametrisation trick
        encoded = means + std * torch.rand_like(std)

        prediction = l.decode(encoded)
        prediction = F.dropout(prediction, p=0.5, training=self.training)

        self.means = means
        self.log_std = log_std

        return prediction
