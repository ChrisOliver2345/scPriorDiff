from __future__ import annotations

import torch
from torch import nn


class ScalarValueEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values.unsqueeze(-1))


class EdgeEncoder(nn.Module):
    """Sum two-state embeddings for the five heterogeneous relations."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.relation_embeddings = nn.ModuleList(
            nn.Embedding(2, hidden_dim) for _ in range(5)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, flags: torch.Tensor) -> torch.Tensor:
        encoded = sum(
            embedding(flags[:, index])
            for index, embedding in enumerate(self.relation_embeddings)
        )
        return self.norm(encoded)


class MessagePassingLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        edge_chunk_size: int,
        message_scale: float = 0.01,
    ):
        super().__init__()
        self.edge_chunk_size = edge_chunk_size
        self.message_scale = message_scale
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_embeddings: torch.Tensor,
        in_degree: torch.Tensor,
    ) -> torch.Tensor:
        aggregated = torch.zeros_like(nodes)
        edge_count = edge_index.shape[1]

        for start in range(0, edge_count, self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, edge_count)
            source = edge_index[0, start:end]
            target = edge_index[1, start:end]
            source_nodes = nodes[:, source]
            target_nodes = nodes[:, target]
            edge_values = edge_embeddings[start:end]

            edge_delta = self.edge_update(torch.cat([
                source_nodes.mean(dim=0),
                edge_values,
                target_nodes.mean(dim=0),
            ], dim=-1))
            edge_values = self.edge_norm(edge_values + self.dropout(edge_delta))
            messages = self.message(torch.cat([
                source_nodes,
                edge_values.unsqueeze(0).expand(nodes.shape[0], -1, -1),
                target_nodes,
            ], dim=-1))
            aggregated.index_add_(1, target, messages)

        aggregated = aggregated / in_degree.view(1, -1, 1)
        return self.node_norm(
            nodes + self.message_scale * self.dropout(aggregated)
        )


class HeterogeneousGraphEncoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_peaks: int,
        latent_dim: int,
        hidden_dim: int,
        edge_index: torch.Tensor,
        edge_flags: torch.Tensor,
        layers: int,
        dropout: float,
        edge_chunk_size: int,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_peaks = n_peaks
        self.n_nodes = 2 + n_genes + n_peaks
        self.identity = nn.Embedding(self.n_nodes, hidden_dim)
        self.identity_norm = nn.LayerNorm(hidden_dim)
        self.value_encoder = ScalarValueEncoder(hidden_dim, dropout)
        self.rna_cell_projection = nn.Linear(latent_dim, hidden_dim)
        self.atac_cell_projection = nn.Linear(latent_dim, hidden_dim)
        self.rna_output = nn.Linear(hidden_dim, latent_dim)
        self.atac_output = nn.Linear(hidden_dim, latent_dim)
        self.edge_encoder = EdgeEncoder(hidden_dim)
        self.layers = nn.ModuleList(
            MessagePassingLayer(
                hidden_dim,
                dropout,
                edge_chunk_size,
            )
            for _ in range(layers)
        )
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_flags", edge_flags)
        degree = torch.bincount(edge_index[1], minlength=self.n_nodes).float().clamp_min(1)
        self.register_buffer("in_degree", degree)

    def identity_tokens(self, modality: str) -> torch.Tensor:
        if modality == "rna":
            indices = torch.arange(2, 2 + self.n_genes, device=self.edge_index.device)
        else:
            start = 2 + self.n_genes
            indices = torch.arange(start, start + self.n_peaks, device=self.edge_index.device)
        return self.identity_norm(self.identity(indices))

    def value_tokens(self, values: torch.Tensor) -> torch.Tensor:
        return self.value_encoder(values)

    def molecular_tokens(
        self,
        rna_values: torch.Tensor,
        atac_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rna = self.identity_tokens("rna").unsqueeze(0) + self.value_tokens(rna_values)
        atac = self.identity_tokens("atac").unsqueeze(0) + self.value_tokens(atac_values)
        return rna, atac

    def forward_tokens(
        self,
        rna_tokens: torch.Tensor,
        atac_tokens: torch.Tensor,
        rna_cell: torch.Tensor | None,
        atac_cell: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = rna_tokens.shape[0]
        dtype = rna_tokens.dtype
        device = rna_tokens.device
        if rna_cell is None:
            rna_cell_node = torch.zeros(batch_size, 1, rna_tokens.shape[-1], device=device, dtype=dtype)
        else:
            rna_cell_node = self.rna_cell_projection(rna_cell).unsqueeze(1)
        if atac_cell is None:
            atac_cell_node = torch.zeros(batch_size, 1, rna_tokens.shape[-1], device=device, dtype=dtype)
        else:
            atac_cell_node = self.atac_cell_projection(atac_cell).unsqueeze(1)

        cell_identity = self.identity_norm(self.identity(torch.arange(2, device=device)))
        rna_cell_node = rna_cell_node + cell_identity[0].view(1, 1, -1)
        atac_cell_node = atac_cell_node + cell_identity[1].view(1, 1, -1)
        nodes = torch.cat([rna_cell_node, atac_cell_node, rna_tokens, atac_tokens], dim=1)
        edge_embeddings = self.edge_encoder(self.edge_flags)
        for layer in self.layers:
            nodes = layer(nodes, self.edge_index, edge_embeddings, self.in_degree)
        rna_graph = self.rna_output(nodes[:, 0])
        atac_graph = self.atac_output(nodes[:, 1])
        return rna_graph, atac_graph
