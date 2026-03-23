"""Entrenador modular para el modelo de TTS."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split

from tts_project.checkpoint import save_checkpoint
from tts_project.dataset import TTSDataset, collate_batch
from tts_project.model import SimpleTTS, masked_l1
from tts_project.text import TextTokenizer


def set_seed(seed: int) -> None:
    """Fija semillas para hacer mas consistente el entrenamiento."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TTSTrainer:
    """Orquesta el ciclo de entrenamiento y validacion."""

    def __init__(self, config: dict, manifest_path: Path, checkpoint_path: Path) -> None:
        self.config = config
        self.manifest_path = manifest_path
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = TextTokenizer()

    def build_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Construye dataloaders de entrenamiento y validacion."""
        dataset = TTSDataset(self.manifest_path, self.config, self.tokenizer)
        if len(dataset) < 2:
            raise ValueError("Se necesitan al menos 2 muestras procesadas para poder entrenar.")

        train_size = max(1, min(len(dataset) - 1, int(len(dataset) * self.config["train_split"])))
        valid_size = len(dataset) - train_size
        train_set, valid_set = random_split(dataset, [train_size, valid_size])
        loader_kwargs = {
            "batch_size": self.config["batch_size"],
            "collate_fn": lambda batch: collate_batch(batch, self.tokenizer.pad_id),
        }
        return (
            DataLoader(train_set, shuffle=True, **loader_kwargs),
            DataLoader(valid_set, shuffle=False, **loader_kwargs),
        )

    def run(self) -> None:
        """Ejecuta el entrenamiento completo y guarda checkpoints."""
        set_seed(self.config["seed"])
        train_loader, valid_loader = self.build_loaders()
        model = SimpleTTS(
            self.tokenizer.vocab_size,
            self.config["hidden_size"],
            self.config["n_mels"],
            self.config["num_layers"],
        ).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.config["learning_rate"])

        for epoch in range(1, self.config["epochs"] + 1):
            train_loss = self.run_epoch(model, train_loader, optimizer)
            valid_loss = self.run_epoch(model, valid_loader, optimizer=None)
            save_checkpoint(
                self.checkpoint_path,
                model,
                self.config,
                self.tokenizer,
                metadata={"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss},
            )
            print(f"Epoch {epoch:03d} | train={train_loss:.4f} | valid={valid_loss:.4f}")

    def run_epoch(self, model: SimpleTTS, loader: DataLoader, optimizer: optim.Optimizer | None) -> float:
        """Ejecuta una epoca de entrenamiento o evaluacion."""
        is_training = optimizer is not None
        model.train(mode=is_training)
        total_loss = 0.0
        for batch in loader:
            if optimizer is not None:
                optimizer.zero_grad()
            outputs = model(batch["tokens"].to(self.device), batch["mel_lengths"].to(self.device))
            mel_loss = masked_l1(outputs["mel"], batch["mels"].to(self.device), batch["mel_lengths"].to(self.device))
            len_loss = torch.nn.functional.l1_loss(outputs["length_prediction"], batch["mel_lengths"].float().to(self.device))
            loss = mel_loss + 0.1 * len_loss
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
        return total_loss / max(1, len(loader))
