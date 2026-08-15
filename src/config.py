from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ogbn_arxiv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class TrainingConfig:
    hidden_channels: int = 256
    num_layers: int = 2
    dropout: float = 0.5
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
