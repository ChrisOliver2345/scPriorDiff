from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ExperimentConfig, TaskSpec
from .attention_unet import AttentionUNet1D
from .diffusion import DiffusionProcess
from .model import Stage1Model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def stage1_epoch(
    model: Stage1Model,
    loader: DataLoader,
    device: torch.device,
    weights: dict,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_cells = 0

    for batch in loader:
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(batch)
            rna_loss = nn.functional.mse_loss(output.rna_reconstruction, batch["rna_full"])
            atac_loss = nn.functional.binary_cross_entropy_with_logits(
                output.atac_logits, batch["atac_full"]
            )
            loss = (
                float(weights["rna_reconstruction"]) * rna_loss
                + float(weights["atac_reconstruction"]) * atac_loss
                + float(weights["completion"]) * output.completion_loss
                + float(weights["proximal_prior"]) * output.prior_loss
            )
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        batch_size = batch["rna_full"].shape[0]
        total_loss += loss.item() * batch_size
        total_cells += batch_size
    return total_loss / total_cells


@torch.no_grad()
def extract_latents(
    model: Stage1Model,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    conditions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in loader:
        batch = move_batch(batch, device)
        conditions.append(model.encode_condition(batch).cpu())
        targets.append(model.encode_target(batch).cpu())
    return torch.cat(conditions), torch.cat(targets)


def build_diffusion(config: ExperimentConfig, latent_dim: int) -> DiffusionProcess:
    settings = config.section("diffusion")
    denoiser = AttentionUNet1D(
        dimension=int(settings["base_channels"]),
        channels=2,
        output_channels=1,
        dimension_multipliers=tuple(settings["unet_dim_mults"]),
    )
    return DiffusionProcess(
        denoiser=denoiser,
        timesteps=int(settings["timesteps"]),
        beta_start=float(settings["beta_start"]),
        beta_end=float(settings["beta_end"]),
    )


def diffusion_epoch(
    diffusion: DiffusionProcess,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    diffusion.train(training)
    total_loss = 0.0
    total_cells = 0
    for condition, target in loader:
        condition = condition.to(device)
        target = target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss = diffusion.training_loss(target, condition)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(diffusion.parameters(), 5.0)
                optimizer.step()
        total_loss += loss.item() * condition.shape[0]
        total_cells += condition.shape[0]
    return total_loss / total_cells


def latent_loader(
    conditions: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(conditions, targets),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def save_history(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def save_checkpoint(path: Path, model: nn.Module, task: TaskSpec, fold: int) -> None:
    torch.save({"state_dict": model.state_dict(), "task": task.name, "fold": fold}, path)


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
