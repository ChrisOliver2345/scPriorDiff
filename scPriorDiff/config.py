from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    source: str
    target: str

    @classmethod
    def from_name(cls, name: str) -> "TaskSpec":
        tasks = {
            "RNA-ATAC": cls("RNA-ATAC", "rna", "atac"),
            "ATAC-RNA": cls("ATAC-RNA", "atac", "rna"),
        }
        return tasks[name]


@dataclass(frozen=True)
class ExperimentConfig:
    root: Path
    values: dict[str, Any]

    @property
    def raw_dir(self) -> Path:
        return self.root / self.values["data"]["raw_dir"]

    @property
    def processed_dir(self) -> Path:
        return self.root / self.values["data"]["processed_dir"]

    @property
    def output_dir(self) -> Path:
        return self.root / self.values["output_dir"]

    def section(self, name: str) -> dict[str, Any]:
        return self.values[name]

    def run_dir(self, task: TaskSpec) -> Path:
        return self.output_dir / task.name


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    return ExperimentConfig(root=PROJECT_ROOT, values=values)
