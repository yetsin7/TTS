"""Dataset y funciones de padding para entrenar TTS."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from tts_project.audio import load_audio, waveform_to_mel
from tts_project.text import TextTokenizer


class TTSDataset(Dataset):
    """Carga un manifiesto JSONL con texto y ruta de audio."""

    def __init__(self, manifest_path: Path, config: dict, tokenizer: TextTokenizer) -> None:
        self.items = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.config = config
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.items[index]
        tokens = torch.tensor(self.tokenizer.encode(item["text"]), dtype=torch.long)
        waveform = load_audio(Path(item["audio_path"]), self.config["sample_rate"])
        mel = waveform_to_mel(waveform, self.config)
        return {"tokens": tokens, "mel": mel}


def collate_batch(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    """Aplica padding para texto y mel en un mini-batch."""
    token_lengths = torch.tensor([item["tokens"].size(0) for item in batch], dtype=torch.long)
    mel_lengths = torch.tensor([item["mel"].size(0) for item in batch], dtype=torch.long)
    tokens = pad_sequence([item["tokens"] for item in batch], batch_first=True, padding_value=pad_id)
    mels = pad_sequence([item["mel"] for item in batch], batch_first=True, padding_value=0.0)
    return {
        "tokens": tokens,
        "token_lengths": token_lengths,
        "mels": mels,
        "mel_lengths": mel_lengths,
    }
