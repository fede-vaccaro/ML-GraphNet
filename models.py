import torch
import torch.nn as nn
import torch.nn.functional as F
import layers as l
import utils as u


class GCNAutoencoder(nn.Module):
    def __init__(self, A, n_features, hidden_dim=32, code_dim=16):
        super().__init__()
        self.hidden = l.KipfAndWillingConv(hidden_dim, n_features)
        self.encoder = l.KipfAndWillingConv(code_dim, hidden_dim)

        transform = l.KipfAndWillingConv.compute_transform(A)
        self.transform = u.dense_to_sparse(transform)

    def forward(self, x):
        hidden = self.hidden(x, self.transform)
        hidden = F.relu(hidden)

        hidden = F.dropout(hidden, p=0.5, training=self.training)

        encoded = self.encoder(hidden, self.transform)
        prediction = l.decode(encoded)

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
        self.log_std2_encoder = l.KipfAndWillingConv(hidden_dim, code_dim)

        transform = l.KipfAndWillingConv.compute_transform(A)

        self.transform = u.dense_to_sparse(transform)
        self.ones = torch.ones(n_samples, code_dim)
        self.n_samples = n_samples

    def to(self, device):
        self.transform = self.transform.to(device)
        self.ones = self.ones.to(device)
        super().to(device)

    def kl_divergence(self):
        mu2 = self.means.pow(2.0)
        logs_plus_m_less_s = self.log_std2 - mu2 - self.std2

        kl = logs_plus_m_less_s + self.ones
        kl = kl.sum(dim=1).mean() * 0.5/self.n_samples

        return -kl

    def forward(self, x):
        hidden = self.hidden(x, self.transform)
        hidden = F.relu(hidden)

        hidden = F.dropout(hidden, p=0.5, training=self.training)

        means = self.means_encoder(hidden, self.transform)
        log_std2 = self.log_std2_encoder(hidden, self.transform)
        std2 = torch.exp(log_std2)

        # reparametrisation trick
        encoded = means + std2 * torch.normal(mean=means)

        prediction = l.decode(encoded)

        self.means = means
        self.log_std2 = log_std2
        self.std2 = std2

        return prediction
