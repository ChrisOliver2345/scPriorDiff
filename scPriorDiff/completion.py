from __future__ import annotations

import math

import torch
from torch import nn


class PriorGuidedCompletion(nn.Module):
    """Task-specific cross-attention with a soft proximal prior."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        prior_edges: torch.Tensor,
        target_count: int,
        source_count: int,
        chunk_size: int,
        dropout: float,
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by attention heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.target_count = target_count
        self.source_count = source_count
        self.chunk_size = chunk_size
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.register_buffer("prior_edges", prior_edges)

    def forward(
        self,
        source_tokens: torch.Tensor,
        target_identity: torch.Tensor,
        target_value_tokens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = source_tokens.shape[0]
        keys = self.key(source_tokens).view(batch_size, self.source_count, self.heads, self.head_dim)
        values = self.value(source_tokens).view(batch_size, self.source_count, self.heads, self.head_dim)
        completed_chunks: list[torch.Tensor] = []
        completion_loss = source_tokens.new_zeros(())
        prior_loss = source_tokens.new_zeros(())
        chunk_count = 0

        prior_target, prior_source = self.prior_edges
        for start in range(0, self.target_count, self.chunk_size):
            end = min(start + self.chunk_size, self.target_count)
            identity = target_identity[start:end]
            queries = self.query(identity).view(end - start, self.heads, self.head_dim)
            logits = torch.einsum("thd,bshd->bhts", queries, keys) / math.sqrt(self.head_dim)
            attention = torch.softmax(logits, dim=-1)
            context = torch.einsum("bhts,bshd->bthd", attention, values)
            predicted_value = self.output(context.reshape(batch_size, end - start, -1))
            completed_chunks.append(identity.unsqueeze(0) + predicted_value)

            if target_value_tokens is not None:
                completion_loss = completion_loss + torch.nn.functional.mse_loss(
                    predicted_value,
                    target_value_tokens[:, start:end],
                )

            prior_mask = torch.zeros(
                end - start,
                self.source_count,
                dtype=torch.bool,
                device=source_tokens.device,
            )
            selected = (prior_target >= start) & (prior_target < end)
            prior_mask[prior_target[selected] - start, prior_source[selected]] = True
            outside_mass = attention.masked_fill(prior_mask.unsqueeze(0).unsqueeze(0), 0).sum(dim=-1)
            prior_loss = prior_loss + outside_mass.mean()
            chunk_count += 1

        return (
            torch.cat(completed_chunks, dim=1),
            completion_loss / chunk_count,
            prior_loss / chunk_count,
        )
