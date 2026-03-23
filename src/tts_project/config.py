"""Utilidades de configuracion para el proyecto TTS."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
CONFIG_PATH = ROOT_DIR / "configs" / "train.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Carga la configuracion YAML del entrenamiento."""
    config_path = path or CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
