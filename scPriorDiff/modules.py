from __future__ import annotations

import torch
from torch import nn


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class RNADecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class ChromosomeATACEncoder(nn.Module):
    def __init__(
        self,
        chromosome_sizes: list[int],
        latent_dim: int,
        chromosome_hidden: int = 32,
    ):
        super().__init__()
        self.chromosome_sizes = chromosome_sizes
        self.encoders = nn.ModuleList(
            nn.Linear(size, chromosome_hidden) for size in chromosome_sizes
        )
        self.projection = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(chromosome_hidden * len(chromosome_sizes), latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(values, self.chromosome_sizes, dim=1)
        encoded = [torch.nn.functional.leaky_relu(layer(chunk), 0.2)
                   for layer, chunk in zip(self.encoders, chunks)]
        return self.projection(torch.cat(encoded, dim=1))


class ChromosomeATACDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        chromosome_sizes: list[int],
        chromosome_hidden: int = 32,
    ):
        super().__init__()
        self.chromosome_sizes = chromosome_sizes
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, chromosome_hidden * len(chromosome_sizes)),
            nn.LeakyReLU(0.2),
        )
        self.heads = nn.ModuleList(
            nn.Linear(chromosome_hidden, size) for size in chromosome_sizes
        )
        self.chromosome_hidden = chromosome_hidden

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.shared(latent)
        chunks = torch.split(hidden, self.chromosome_hidden, dim=1)
        return torch.cat([head(chunk) for head, chunk in zip(self.heads, chunks)], dim=1)
