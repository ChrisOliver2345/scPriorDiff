from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def _default(value, fallback):
    if value is not None:
        return value
    return fallback() if callable(fallback) else fallback


class Residual(nn.Module):
    def __init__(self, function: nn.Module):
        super().__init__()
        self.function = function

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.function(values) + values


def upsample(input_dim: int, output_dim: int | None = None) -> nn.Module:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv1d(input_dim, _default(output_dim, input_dim), 3, padding=1),
    )


def downsample(input_dim: int, output_dim: int | None = None) -> nn.Module:
    return nn.Conv1d(input_dim, _default(output_dim, input_dim), 4, 2, 1)


class WeightStandardizedConv1d(nn.Conv1d):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-5 if values.dtype == torch.float32 else 1e-3
        mean = self.weight.mean(dim=(1, 2), keepdim=True)
        variance = self.weight.var(dim=(1, 2), unbiased=False, keepdim=True)
        weight = (self.weight - mean) * (variance + epsilon).rsqrt()
        return F.conv1d(
            values,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ChannelLayerNorm(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, dimension, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-5 if values.dtype == torch.float32 else 1e-3
        variance = values.var(dim=1, unbiased=False, keepdim=True)
        mean = values.mean(dim=1, keepdim=True)
        return (values - mean) * (variance + epsilon).rsqrt() * self.scale


class PreNorm(nn.Module):
    def __init__(self, dimension: int, function: nn.Module):
        super().__init__()
        self.norm = ChannelLayerNorm(dimension)
        self.function = function

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.function(self.norm(values))


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half_dimension = self.dimension // 2
        scale = math.log(10_000) / (half_dimension - 1)
        frequencies = torch.exp(
            torch.arange(half_dimension, device=time.device) * -scale
        )
        angles = time[:, None] * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class Block(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, groups: int = 8):
        super().__init__()
        self.projection = WeightStandardizedConv1d(
            input_dim, output_dim, 3, padding=1
        )
        self.norm = nn.GroupNorm(groups, output_dim)
        self.activation = nn.SiLU()

    def forward(
        self,
        values: torch.Tensor,
        scale_shift: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        values = self.projection(values)
        values = self.norm(values)
        if scale_shift is not None:
            scale, shift = scale_shift
            values = values * (scale + 1) + shift
        return self.activation(values)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        time_dim: int,
        groups: int = 8,
    ):
        super().__init__()
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, output_dim * 2))
        self.block1 = Block(input_dim, output_dim, groups=groups)
        self.block2 = Block(output_dim, output_dim, groups=groups)
        self.residual = (
            nn.Conv1d(input_dim, output_dim, 1)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, values: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        scale_shift = rearrange(self.time(time), "b c -> b c 1").chunk(2, dim=1)
        hidden = self.block1(values, scale_shift=scale_shift)
        hidden = self.block2(hidden)
        return hidden + self.residual(values)


class LinearAttention(nn.Module):
    def __init__(self, dimension: int, heads: int = 4, head_dim: int = 32):
        super().__init__()
        self.scale = head_dim**-0.5
        self.heads = heads
        hidden_dim = head_dim * heads
        self.to_qkv = nn.Conv1d(dimension, hidden_dim * 3, 1, bias=False)
        self.output = nn.Sequential(
            nn.Conv1d(hidden_dim, dimension, 1),
            ChannelLayerNorm(dimension),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, _, sequence_length = values.shape
        query, key, value = (
            rearrange(tensor, "b (h c) n -> b h c n", h=self.heads)
            for tensor in self.to_qkv(values).chunk(3, dim=1)
        )
        query = query.softmax(dim=-2) * self.scale
        key = key.softmax(dim=-1)
        value = value / sequence_length
        context = torch.einsum("b h d n, b h e n -> b h d e", key, value)
        output = torch.einsum("b h d e, b h d n -> b h e n", context, query)
        output = rearrange(output, "b h c n -> b (h c) n", h=self.heads)
        return self.output(output)


class Attention(nn.Module):
    def __init__(self, dimension: int, heads: int = 4, head_dim: int = 32):
        super().__init__()
        self.scale = head_dim**-0.5
        self.heads = heads
        hidden_dim = head_dim * heads
        self.to_qkv = nn.Conv1d(dimension, hidden_dim * 3, 1, bias=False)
        self.output = nn.Conv1d(hidden_dim, dimension, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        query, key, value = (
            rearrange(tensor, "b (h c) n -> b h c n", h=self.heads)
            for tensor in self.to_qkv(values).chunk(3, dim=1)
        )
        query = query * self.scale
        similarity = torch.einsum("b h d i, b h d j -> b h i j", query, key)
        weights = similarity.softmax(dim=-1)
        output = torch.einsum("b h i j, b h d j -> b h i d", weights, value)
        output = rearrange(output, "b h n d -> b (h d) n")
        return self.output(output)


class AttentionUNet1D(nn.Module):
    def __init__(
        self,
        dimension: int,
        channels: int = 2,
        output_channels: int = 1,
        dimension_multipliers: tuple[int, ...] = (1, 2, 4, 8, 8),
        groups: int = 8,
    ):
        super().__init__()
        initial_dim = dimension
        self.input = nn.Conv1d(channels, initial_dim, 7, padding=3)
        dimensions = [initial_dim, *[dimension * value for value in dimension_multipliers]]
        dimension_pairs = list(zip(dimensions[:-1], dimensions[1:]))
        block = partial(ResnetBlock, groups=groups)
        time_dim = dimension * 4

        self.time = nn.Sequential(
            SinusoidalPositionEmbedding(dimension),
            nn.Linear(dimension, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.down_blocks = nn.ModuleList()
        for index, (input_dim, output_dim) in enumerate(dimension_pairs):
            last = index == len(dimension_pairs) - 1
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        block(input_dim, input_dim, time_dim=time_dim),
                        block(input_dim, input_dim, time_dim=time_dim),
                        Residual(PreNorm(input_dim, LinearAttention(input_dim))),
                        (
                            nn.Conv1d(input_dim, output_dim, 3, padding=1)
                            if last
                            else downsample(input_dim, output_dim)
                        ),
                    ]
                )
            )

        middle_dim = dimensions[-1]
        self.middle1 = block(middle_dim, middle_dim, time_dim=time_dim)
        self.middle_attention = Residual(
            PreNorm(middle_dim, Attention(middle_dim))
        )
        self.middle2 = block(middle_dim, middle_dim, time_dim=time_dim)

        self.up_blocks = nn.ModuleList()
        for index, (input_dim, output_dim) in enumerate(reversed(dimension_pairs)):
            last = index == len(dimension_pairs) - 1
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        block(output_dim + input_dim, output_dim, time_dim=time_dim),
                        block(output_dim + input_dim, output_dim, time_dim=time_dim),
                        Residual(PreNorm(output_dim, LinearAttention(output_dim))),
                        (
                            nn.Conv1d(output_dim, input_dim, 3, padding=1)
                            if last
                            else upsample(output_dim, input_dim)
                        ),
                    ]
                )
            )

        self.final_block = block(dimension * 2, dimension, time_dim=time_dim)
        self.output = nn.Conv1d(dimension, output_channels, 1)

    def forward(self, values: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        values = self.input(values)
        residual = values.clone()
        time_embedding = self.time(time)
        skips: list[torch.Tensor] = []

        for block1, block2, attention, reduce in self.down_blocks:
            values = block1(values, time_embedding)
            skips.append(values)
            values = block2(values, time_embedding)
            values = attention(values)
            skips.append(values)
            values = reduce(values)

        values = self.middle1(values, time_embedding)
        values = self.middle_attention(values)
        values = self.middle2(values, time_embedding)

        for block1, block2, attention, expand in self.up_blocks:
            values = block1(torch.cat((values, skips.pop()), dim=1), time_embedding)
            values = block2(torch.cat((values, skips.pop()), dim=1), time_embedding)
            values = attention(values)
            values = expand(values)

        values = self.final_block(
            torch.cat((values, residual), dim=1), time_embedding
        )
        return self.output(values)
