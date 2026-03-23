"""Servicio reutilizable para sintetizar voz con el modelo entrenado."""

from __future__ import annotations

from pathlib import Path

import torch

from tts_project.audio import mel_to_waveform, save_audio
from tts_project.checkpoint import build_model_from_checkpoint
from tts_project.config import MODELS_DIR, OUTPUTS_DIR
from tts_project.text import TextTokenizer


class TTSService:
    """Envuelve la carga del modelo y la sintesis para CLI, UI y API."""

    def __init__(self, checkpoint_path: Path | None = None, device: str | None = None) -> None:
        self.checkpoint_path = checkpoint_path or MODELS_DIR / "tts_model.pt"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = TextTokenizer()
        self.model = None
        self.config: dict | None = None

    def load(self) -> None:
        """Carga checkpoint y deja el modelo listo para inferencia."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"No existe el checkpoint: {self.checkpoint_path}")
        self.model, checkpoint = build_model_from_checkpoint(self.checkpoint_path, device=self.device)
        symbols = checkpoint.get("symbols")
        if symbols:
            self.tokenizer = TextTokenizer(symbols=symbols)
        self.config = checkpoint["config"]

    def synthesize_to_waveform(self, text: str) -> torch.Tensor:
        """Genera un waveform en memoria a partir de un texto."""
        if self.model is None or self.config is None:
            self.load()
        tokens = torch.tensor([self.tokenizer.encode(text)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            mel = self.model(tokens)["mel"].squeeze(0).cpu()
        return mel_to_waveform(mel, self.config)

    def synthesize_to_file(self, text: str, output_path: Path | None = None) -> Path:
        """Genera un archivo WAV y devuelve su ruta."""
        if self.config is None:
            self.load()
        destination = output_path or (OUTPUTS_DIR / "generated" / "tts_output.wav")
        waveform = self.synthesize_to_waveform(text)
        save_audio(destination, waveform, self.config["sample_rate"])
        return destination
