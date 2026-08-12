from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .completion import PriorGuidedCompletion
from .config import ExperimentConfig, TaskSpec
from .data import GraphData, PreparedData
from .graph import HeterogeneousGraphEncoder
from .modules import (
    ChromosomeATACDecoder,
    ChromosomeATACEncoder,
    MLPEncoder,
    RNADecoder,
)


@dataclass
class Stage1Output:
    rna_reconstruction: torch.Tensor
    atac_logits: torch.Tensor
    completion_loss: torch.Tensor
    prior_loss: torch.Tensor


class Stage1Model(nn.Module):
    def __init__(
        self,
        task: TaskSpec,
        rna_dim: int,
        chromosome_sizes: list[int],
        n_graph_genes: int,
        n_graph_peaks: int,
        graph_data: GraphData,
        model_config: dict,
    ):
        super().__init__()
        self.task = task
        latent_dim = int(model_config["latent_dim"])
        hidden_dim = int(model_config["graph_hidden_dim"])
        self.rna_encoder = MLPEncoder(rna_dim, latent_dim)
        self.atac_encoder = ChromosomeATACEncoder(chromosome_sizes, latent_dim)
        self.rna_decoder = RNADecoder(latent_dim, rna_dim)
        self.atac_decoder = ChromosomeATACDecoder(latent_dim, chromosome_sizes)
        self.graph = HeterogeneousGraphEncoder(
            n_genes=n_graph_genes,
            n_peaks=n_graph_peaks,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            edge_index=graph_data.edge_index,
            edge_flags=graph_data.edge_flags,
            layers=int(model_config["graph_layers"]),
            dropout=float(model_config["dropout"]),
            edge_chunk_size=int(model_config["edge_chunk_size"]),
        )
        prior = (
            graph_data.rna_to_atac_prior
            if task.name == "RNA-ATAC"
            else graph_data.atac_to_rna_prior
        )
        target_count = n_graph_peaks if task.target == "atac" else n_graph_genes
        source_count = n_graph_genes if task.source == "rna" else n_graph_peaks
        self.completion = PriorGuidedCompletion(
            hidden_dim=hidden_dim,
            heads=int(model_config["attention_heads"]),
            prior_edges=prior,
            target_count=target_count,
            source_count=source_count,
            chunk_size=int(model_config["attention_chunk_size"]),
            dropout=float(model_config["dropout"]),
        )

    def _base_latents(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rna_encoder(batch["rna_full"]), self.atac_encoder(batch["atac_full"])

    def _real_latents(
        self,
        batch: dict[str, torch.Tensor],
        rna_base: torch.Tensor,
        atac_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rna_tokens, atac_tokens = self.graph.molecular_tokens(batch["rna_graph"], batch["atac_graph"])
        rna_latent, atac_latent = self.graph.forward_tokens(
            rna_tokens, atac_tokens, rna_base, atac_base
        )
        return rna_latent, atac_latent, rna_tokens, atac_tokens

    def _completed_latents(
        self,
        rna_tokens: torch.Tensor,
        atac_tokens: torch.Tensor,
        rna_base: torch.Tensor,
        atac_base: torch.Tensor,
        supervise: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.task.name == "RNA-ATAC":
            real_target_values = atac_tokens - self.graph.identity_tokens("atac").unsqueeze(0)
            completed, completion_loss, prior_loss = self.completion(
                rna_tokens,
                self.graph.identity_tokens("atac"),
                real_target_values if supervise else None,
            )
            rna_latent, atac_latent = self.graph.forward_tokens(
                rna_tokens, completed, rna_base, None
            )
        else:
            real_target_values = rna_tokens - self.graph.identity_tokens("rna").unsqueeze(0)
            completed, completion_loss, prior_loss = self.completion(
                atac_tokens,
                self.graph.identity_tokens("rna"),
                real_target_values if supervise else None,
            )
            rna_latent, atac_latent = self.graph.forward_tokens(
                completed, atac_tokens, None, atac_base
            )
        return rna_latent, atac_latent, completion_loss, prior_loss

    def forward(self, batch: dict[str, torch.Tensor]) -> Stage1Output:
        rna_base, atac_base = self._base_latents(batch)
        real_rna, real_atac, rna_tokens, atac_tokens = self._real_latents(
            batch, rna_base, atac_base
        )
        _, _, completion_loss, prior_loss = self._completed_latents(
            rna_tokens, atac_tokens, rna_base, atac_base, supervise=True
        )
        return Stage1Output(
            rna_reconstruction=self.rna_decoder(real_rna),
            atac_logits=self.atac_decoder(real_atac),
            completion_loss=completion_loss,
            prior_loss=prior_loss,
        )

    @torch.no_grad()
    def encode_condition(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.task.source == "rna":
            rna_base = self.rna_encoder(batch["rna_full"])
            rna_tokens = (
                self.graph.identity_tokens("rna").unsqueeze(0)
                + self.graph.value_tokens(batch["rna_graph"])
            )
            atac_tokens = self.graph.identity_tokens("atac").unsqueeze(0).expand(
                batch["rna_graph"].shape[0], -1, -1
            )
            condition, _, _, _ = self._completed_latents(
                rna_tokens, atac_tokens, rna_base, rna_base, supervise=False
            )
        else:
            atac_base = self.atac_encoder(batch["atac_full"])
            atac_tokens = (
                self.graph.identity_tokens("atac").unsqueeze(0)
                + self.graph.value_tokens(batch["atac_graph"])
            )
            rna_tokens = self.graph.identity_tokens("rna").unsqueeze(0).expand(
                batch["atac_graph"].shape[0], -1, -1
            )
            _, condition, _, _ = self._completed_latents(
                rna_tokens, atac_tokens, atac_base, atac_base, supervise=False
            )
        return condition

    @torch.no_grad()
    def encode_target(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        rna_base, atac_base = self._base_latents(batch)
        real_rna, real_atac, _, _ = self._real_latents(batch, rna_base, atac_base)
        return real_atac if self.task.target == "atac" else real_rna

    def decode_target(self, latent: torch.Tensor) -> torch.Tensor:
        if self.task.target == "atac":
            return torch.sigmoid(self.atac_decoder(latent))
        return self.rna_decoder(latent)


def build_stage1_model(
    config: ExperimentConfig,
    task: TaskSpec,
    data: PreparedData,
    graph_data: GraphData,
) -> Stage1Model:
    return Stage1Model(
        task=task,
        rna_dim=data.rna_full.shape[1],
        chromosome_sizes=data.chromosome_sizes,
        n_graph_genes=data.rna_graph.shape[1],
        n_graph_peaks=data.atac_graph.shape[1],
        graph_data=graph_data,
        model_config=config.section("model"),
    )
