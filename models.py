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


class DVNE(nn.Module):
    def __init__(self, n_features, n_samples, hidden_dim=512, code_dim=128):
        super().__init__()
        self.hidden = nn.Linear(in_features=n_features, out_features=hidden_dim)

        self.means = nn.Linear(in_features=hidden_dim, out_features=code_dim)
        self.std = nn.Linear(in_features=hidden_dim, out_features=code_dim)

        self.decoder = nn.Linear(in_features=code_dim, out_features=hidden_dim)
        self.output = nn.Linear(in_features=hidden_dim, out_features=n_features)


    def wasserstein(self, n_a, n_b):
        m_a, std_a = n_a
        m_b, std_b = n_b

        dist_m = (m_a - m_b).pow(2.0).sum(dim=1)
        dist_std = (std_a - std_b).pow(2.0).sum(dim=1)
        # same as
        # dist_std = (std_a - std_b).norm(p='fro', dim=1).pow(2.0)

        return (dist_m + dist_std)

    def decode(self, x):
        hidden = F.relu(self.hidden(x))
        #hidden = F.dropout(hidden, p=0.5, training=self.training)

        means = self.means(hidden)
        std = F.elu(self.std(hidden)) + 1

        # reparametrisation trick
        encoded = means + std * torch.rand_like(std)
        return encoded, means, std

    def forward(self, x):
        encoded, means, std = self.decode(x)
        #encoded = F.dropout(encoded, p=0.5, training=self.training)

        decoded = F.relu(self.decoder(encoded))
        #decoded = F.dropout(decoded, p=0.5, training=self.training)

        output = self.output(decoded)
        output = torch.relu(output)

        prediction = output
        return prediction, means, std

class DvneGCN(nn.Module):
    def __init__(self, A, n_features, n_samples, hidden_dim=32, code_dim=16):
        super().__init__()
        self.hidden = l.KipfAndWillingConv(n_features, hidden_dim)

        self.means_encoder = l.KipfAndWillingConv(hidden_dim, code_dim)
        self.log_std_encoder = l.KipfAndWillingConv(hidden_dim, code_dim)

        self.decoder = l.KipfAndWillingConv(code_dim, hidden_dim)
        self.output = l.KipfAndWillingConv(hidden_dim, n_features)

        transform = l.KipfAndWillingConv.compute_transform(A)
        self.transform = u.dense_to_sparse(transform)

    def wasserstein(self, n_a, n_b):
        m_a, std_a = n_a
        m_b, std_b = n_b

        dist_m = (m_a - m_b).pow(2.0).sum(dim=1)
        dist_std = (std_a - std_b).pow(2.0).sum(dim=1)
        # same as
        # dist_std = (std_a - std_b).norm(p='fro', dim=1).pow(2.0)

        return (dist_m + dist_std)

    def decode(self, x):
        hidden = F.relu(self.hidden(x, self.transform))
        #hidden = F.dropout(hidden, p=0.5, training=self.training)

        means = self.means(hidden, self.transform)
        std = F.elu(self.std(hidden, self.transform)) + 1

        # reparametrisation trick
        encoded = means + std * torch.rand_like(std)
        return encoded, means, std

    def forward(self, x):
        encoded, means, std = self.decode(x)
        #encoded = F.dropout(encoded, p=0.5, training=self.training)

        decoded = F.relu(self.decoder(encoded, self.transform))
        #decoded = F.dropout(decoded, p=0.5, training=self.training)

        output = self.output(decoded, self.transform)
        output = torch.relu(output)

        prediction = output
        return prediction, means, std


class GcnVAE(nn.Module):
    def __init__(self, A, n_features, n_samples, hidden_dim=32, code_dim=16):
        super().__init__()
        self.hidden = l.KipfAndWillingConv(n_features, hidden_dim)

        self.means_encoder = l.KipfAndWillingConv(hidden_dim, code_dim)
        self.log_std_encoder = l.KipfAndWillingConv(hidden_dim, code_dim)

        transform = l.KipfAndWillingConv.compute_transform(A)

        self.transform = u.dense_to_sparse(transform)
        self.n_samples = n_samples

    def to(self, device):
        self.transform = self.transform.to(device)
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

        # encoded = F.normalize(encoded, dim=1, p=2)
        prediction = l.decode(encoded)
        prediction = F.dropout(prediction, p=0.5, training=self.training)

        self.means = means
        self.log_std = log_std

        return prediction
