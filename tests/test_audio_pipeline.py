"""Pruebas del pipeline de audio para preparacion de corpus."""

from __future__ import annotations

import torch

from tts_project.audio import segment_audio_for_transcript, split_evenly_by_count, trim_silence


def test_trim_silence_recorta_extremos() -> None:
    """Verifica que se eliminen silencios de los extremos."""
    waveform = torch.tensor([[0.0, 0.0, 0.2, 0.3, 0.0]])
    trimmed = trim_silence(waveform, threshold=0.05)
    assert trimmed.shape[-1] == 2


def test_split_evenly_respeta_cantidad() -> None:
    """Verifica que la division uniforme genere la cantidad esperada."""
    waveform = torch.ones(1, 100)
    clips = split_evenly_by_count(waveform, expected_count=4)
    assert len(clips) == 4


def test_segment_audio_usa_respaldo_si_falla_silencio() -> None:
    """Verifica que exista una estrategia de respaldo estable."""
    waveform = torch.ones(1, 1000)
    clips, strategy = segment_audio_for_transcript(waveform, 100, 3, 200, 0.5, 500)
    assert len(clips) == 3
    assert strategy == "even_split_fallback"
