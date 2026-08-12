from __future__ import annotations

import torch
from torch import nn

from .attention_unet import AttentionUNet1D


class DiffusionProcess(nn.Module):
    def __init__(
        self,
        denoiser: AttentionUNet1D,
        timesteps: int,
        beta_start: float,
        beta_end: float,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)

    def predict_noise(
        self,
        noisy_target: torch.Tensor,
        condition: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.stack((noisy_target, condition), dim=1)
        return self.denoiser(inputs, time).squeeze(1)

    def training_loss(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        time = torch.randint(
            0, self.timesteps, (target.shape[0],), device=target.device
        )
        noise = torch.randn_like(target)
        alpha_bar = self.alpha_bars[time].unsqueeze(1)
        noisy_target = (
            alpha_bar.sqrt() * target
            + (1 - alpha_bar).sqrt() * noise
        )
        predicted_noise = self.predict_noise(noisy_target, condition, time)
        return torch.nn.functional.mse_loss(predicted_noise, noise)

    def sampling_parameters(
        self,
        steps: int,
        eta: float,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        betas = torch.linspace(
            self.beta_start, self.beta_end, steps, device=device
        )
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        previous_alpha_bars = torch.cat(
            (torch.ones(1, device=device), alpha_bars[:-1])
        )
        sigmas = eta * torch.sqrt(
            ((1 - previous_alpha_bars) / (1 - alpha_bars))
            * (1 - alpha_bars / previous_alpha_bars)
        )
        return {
            "alpha_bars": alpha_bars,
            "previous_alpha_bars": previous_alpha_bars,
            "sqrt_alpha_bars": alpha_bars.sqrt(),
            "sqrt_one_minus_alpha_bars": (1 - alpha_bars).sqrt(),
            "sigmas": sigmas,
        }

    def reverse_step(
        self,
        noisy_target: torch.Tensor,
        condition: torch.Tensor,
        time: torch.Tensor,
        parameters: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        index = int(time[0].item())
        sqrt_alpha_bar = parameters["sqrt_alpha_bars"][index]
        sqrt_one_minus_alpha_bar = parameters[
            "sqrt_one_minus_alpha_bars"
        ][index]
        sigma = parameters["sigmas"][index]

        predicted_noise = self.predict_noise(noisy_target, condition, time)
        predicted_clean = (
            noisy_target - sqrt_one_minus_alpha_bar * predicted_noise
        ) / sqrt_alpha_bar
        if index == 0:
            return predicted_clean

        previous_target = (
            parameters["sqrt_alpha_bars"][index - 1] * predicted_clean
            + parameters["sqrt_one_minus_alpha_bars"][index - 1]
            * predicted_noise
        )
        if sigma.item() > 0:
            previous_target = previous_target + sigma * torch.randn_like(
                noisy_target
            )
        return previous_target

    def released_reverse_step(
        self,
        noisy_condition: torch.Tensor,
        noisy_target: torch.Tensor,
        time: torch.Tensor,
        parameters: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        previous_target = self.reverse_step(
            noisy_target, noisy_condition, time, parameters
        )
        previous_condition = self.reverse_step(
            noisy_condition, noisy_target, time, parameters
        )
        return previous_condition, previous_target

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        steps: int,
        samples: int,
        eta: float = 0.0,
        release_time: int = 0,
    ) -> torch.Tensor:
        parameters = self.sampling_parameters(steps, eta, condition.device)
        cell_count = condition.shape[0]
        expanded_condition = condition.repeat(samples, 1)
        noisy_target = torch.randn_like(expanded_condition)

        for index in reversed(range(steps)):
            time = torch.full(
                (expanded_condition.shape[0],),
                index,
                device=condition.device,
                dtype=torch.long,
            )
            if index > release_time:
                noisy_target = self.reverse_step(
                    noisy_target, expanded_condition, time, parameters
                )
            else:
                noise = torch.randn_like(expanded_condition)
                noisy_condition = (
                    parameters["sqrt_alpha_bars"][index]
                    * expanded_condition
                    + parameters["sqrt_one_minus_alpha_bars"][index]
                    * noise
                )
                _, noisy_target = self.released_reverse_step(
                    noisy_condition,
                    noisy_target,
                    time,
                    parameters,
                )

        return noisy_target.reshape(samples, cell_count, -1).mean(dim=0)
