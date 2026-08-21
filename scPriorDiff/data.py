from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class PreparedData:
    rna_full: sp.csr_matrix
    atac_full: sp.csr_matrix
    rna_graph: sp.csr_matrix
    atac_graph: sp.csr_matrix
    cells: pd.DataFrame
    rna_features: pd.DataFrame
    atac_features: pd.DataFrame
    graph_genes: pd.DataFrame
    graph_peaks: pd.DataFrame

    @classmethod
    def load(cls, data_dir: Path) -> "PreparedData":
        return cls(
            rna_full=sp.load_npz(data_dir / "rna_full.npz").tocsr(),
            atac_full=sp.load_npz(data_dir / "atac_full.npz").tocsr(),
            rna_graph=sp.load_npz(data_dir / "rna_graph.npz").tocsr(),
            atac_graph=sp.load_npz(data_dir / "atac_graph.npz").tocsr(),
            cells=pd.read_csv(data_dir / "cells.csv"),
            rna_features=pd.read_csv(data_dir / "rna_features.csv"),
            atac_features=pd.read_csv(data_dir / "atac_features.csv"),
            graph_genes=pd.read_csv(data_dir / "graph_genes.csv"),
            graph_peaks=pd.read_csv(data_dir / "graph_peaks.csv"),
        )

    @property
    def chromosome_sizes(self) -> list[int]:
        counts = self.atac_features.groupby("chrom", sort=False).size()
        return counts.astype(int).tolist()


@dataclass
class GraphData:
    edge_index: torch.Tensor
    edge_flags: torch.Tensor
    rna_to_atac_prior: torch.Tensor
    atac_to_rna_prior: torch.Tensor

    @classmethod
    def load(cls, data_dir: Path) -> "GraphData":
        arrays = np.load(data_dir / "graph.npz")
        return cls(
            edge_index=torch.from_numpy(arrays["edge_index"]).long(),
            edge_flags=torch.from_numpy(arrays["edge_flags"]).long(),
            rna_to_atac_prior=torch.from_numpy(arrays["rna_to_atac_prior"]).long(),
            atac_to_rna_prior=torch.from_numpy(arrays["atac_to_rna_prior"]).long(),
        )


class PairedRows(Dataset):
    def __init__(self, data: PreparedData, indices: np.ndarray):
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def _row(matrix: sp.csr_matrix, index: int) -> torch.Tensor:
        values = matrix.getrow(index).toarray().ravel().astype(np.float32)
        return torch.from_numpy(values)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        return {
            "index": torch.tensor(index, dtype=torch.long),
            "rna_full": self._row(self.data.rna_full, index),
            "atac_full": self._row(self.data.atac_full, index),
            "rna_graph": self._row(self.data.rna_graph, index),
            "atac_graph": self._row(self.data.atac_graph, index),
        }


def make_loader(
    data: PreparedData,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        PairedRows(data, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def load_partitions(data_dir: Path) -> dict[str, np.ndarray]:
    arrays = np.load(data_dir / "partitions.npz")
    return {name: arrays[name].astype(np.int64) for name in ("train", "test")}


def load_manifest(data_dir: Path) -> dict:
    with (data_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)
