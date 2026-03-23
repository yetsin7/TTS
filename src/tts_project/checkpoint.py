"""Utilidades para guardar y cargar checkpoints del modelo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from tts_project.model import SimpleTTS
from tts_project.text import TextTokenizer


def save_checkpoint(
    path: Path,
    model: SimpleTTS,
    config: dict[str, Any],
    tokenizer: TextTokenizer,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Guarda el estado minimo necesario para inferencia y reentrenamiento."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "vocab_size": tokenizer.vocab_size,
            "symbols": tokenizer.symbols,
            "metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Carga un checkpoint desde disco."""
    return torch.load(path, map_location=device)


def build_model_from_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[SimpleTTS, dict[str, Any]]:
    """Reconstruye el modelo y devuelve tambien la configuracion."""
    checkpoint = load_checkpoint(path, device=device)
    config = checkpoint["config"]
    tokenizer = TextTokenizer(symbols=checkpoint.get("symbols", TextTokenizer().symbols))
    model = SimpleTTS(
        tokenizer.vocab_size,
        config["hidden_size"],
        config["n_mels"],
        config["num_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
