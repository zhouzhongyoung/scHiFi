from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DispAct, MeanAct, ZINBLoss


class SparseGraphConv(nn.Module):
    """Original scHiFi graph layer implemented with sparse matrix multiplication."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, active: bool = True) -> torch.Tensor:
        support = F.linear(x, self.weight)
        if active:
            support = torch.tanh(support)
        return torch.sparse.mm(adj, support)


class MultiScaleSparseGCN(nn.Module):
    def __init__(
        self,
        graph_dim: int,
        hidden_dims: Sequence[int],
        n_clusters: int,
    ):
        super().__init__()
        if len(hidden_dims) != 3:
            raise ValueError("hidden_dims must contain [512, 256, 128]-style dimensions.")
        h1, h2, h3 = map(int, hidden_dims)
        self.gnn1 = SparseGraphConv(graph_dim, h1)
        self.gnn2 = SparseGraphConv(h1, h2)
        self.gnn3 = SparseGraphConv(h2, h3)
        self.gnn4 = SparseGraphConv(h3, n_clusters)

    def forward(
        self, x_graph: torch.Tensor, adjs: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(adjs) != 4:
            raise ValueError("Four scale-specific adjacency matrices are required.")
        g1 = self.gnn1(x_graph, adjs[0], active=True)
        g2 = self.gnn2(g1, adjs[1], active=True)
        g3 = self.gnn3(g2, adjs[2], active=True)
        g4 = self.gnn4(g3, adjs[3], active=True)
        return g1, g2, g3, g4


class ScHiFi(nn.Module):
    def __init__(
        self,
        input_dim: int,
        graph_dim: int,
        zinb_dim: int,
        n_clusters: int,
        z_dim: int = 32,
        encoder_dims: Sequence[int] = (512, 256, 128),
        decoder_dims: Sequence[int] = (128, 256),
        sigma: float = 1.5,
        alpha: float = 1.0,
    ):
        super().__init__()
        h1, h2, h3 = map(int, encoder_dims)
        self.z_dim = int(z_dim)
        self.n_clusters = int(n_clusters)
        self.sigma = float(sigma)
        self.alpha = float(alpha)

        self.gnn = MultiScaleSparseGCN(graph_dim, encoder_dims, n_clusters)
        self.attr1 = nn.Linear(input_dim, h1)
        self.attr2 = nn.Linear(h1, h2)
        self.attr3 = nn.Linear(h2, h3)
        self.enc_mu = nn.Linear(h3, z_dim)

        decoder = []
        dims = [z_dim, *map(int, decoder_dims)]
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            decoder.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        self.decoder = nn.Sequential(*decoder)
        last_dec = int(decoder_dims[-1])
        self.dec_mean = nn.Sequential(nn.Linear(last_dec, zinb_dim), MeanAct())
        self.dec_disp = nn.Sequential(nn.Linear(last_dec, zinb_dim), DispAct())
        self.dec_pi = nn.Sequential(nn.Linear(last_dec, zinb_dim), nn.Sigmoid())

        self.zinb_loss = ZINBLoss()
        self.register_parameter("mu", None)

    def initialize_cluster_centers(self, centers: torch.Tensor) -> None:
        if centers.shape != (self.n_clusters, self.z_dim):
            raise ValueError(
                f"Expected centers {(self.n_clusters, self.z_dim)}, got {tuple(centers.shape)}"
            )
        self.mu = nn.Parameter(centers.detach().clone())

    def encode_topology(
        self, x_graph: torch.Tensor, adjs: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.gnn(x_graph, adjs)

    def encode_attribute(
        self,
        x_attr: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        g3: torch.Tensor,
        add_noise: bool = True,
    ) -> torch.Tensor:
        if add_noise and self.training and self.sigma > 0:
            x_attr = x_attr + torch.randn_like(x_attr) * self.sigma
        h1 = F.relu(self.attr1(x_attr))
        h2 = F.relu(self.attr2(h1 + g1))
        h3 = F.relu(self.attr3(h2 + g2))
        return self.enc_mu(h3 + g3)

    def decode_zinb(self, z: torch.Tensor):
        h = self.decoder(z)
        return self.dec_mean(h), self.dec_disp(h), self.dec_pi(h)

    def soft_assign(self, z: torch.Tensor) -> torch.Tensor:
        if self.mu is None:
            raise RuntimeError("Cluster centers have not been initialized.")
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        return q / torch.clamp(q.sum(dim=1, keepdim=True), min=1e-12)

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        weight = q.square() / torch.clamp(q.sum(dim=0), min=1e-12)
        return weight / torch.clamp(weight.sum(dim=1, keepdim=True), min=1e-12)
