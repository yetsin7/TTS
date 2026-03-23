"""Modelo compacto de TTS para una sola voz."""

from __future__ import annotations

import torch
from torch import nn


class SimpleTTS(nn.Module):
    """Modelo base que interpola embeddings de texto hacia el largo del mel."""

    def __init__(self, vocab_size: int, hidden_size: int, n_mels: int, num_layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Softplus(),
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.mel_head = nn.Linear(hidden_size, n_mels)

    def forward(self, tokens: torch.Tensor, target_lengths: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        embedded = self.embedding(tokens)
        encoded, _ = self.encoder(embedded)
        duration_logits = self.duration_head(encoded).squeeze(-1)
        length_prediction = duration_logits.sum(dim=1).clamp_min(1.0)
        if target_lengths is None:
            target_lengths = length_prediction.round().long()

        upsampled = []
        max_length = int(target_lengths.max().item())
        for sample, mel_length in zip(encoded, target_lengths.tolist(), strict=False):
            sample = sample.transpose(0, 1).unsqueeze(0)
            resized = torch.nn.functional.interpolate(
                sample,
                size=max(1, mel_length),
                mode="linear",
                align_corners=False,
            )
            if mel_length < max_length:
                padding = max_length - mel_length
                resized = torch.nn.functional.pad(resized, (0, padding))
            upsampled.append(resized.squeeze(0).transpose(0, 1))

        decoder_input = torch.stack(upsampled, dim=0)
        decoded, _ = self.decoder(decoder_input)
        mel = self.mel_head(decoded)
        return {"mel": mel, "length_prediction": length_prediction}


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Calcula una perdida L1 ignorando el padding."""
    mask = torch.arange(target.size(1), device=target.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1)
    diff = (prediction - target).abs() * mask
    return diff.sum() / mask.sum().clamp_min(1)
