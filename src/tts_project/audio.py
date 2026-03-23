"""Herramientas de audio para preparar y reconstruir muestras."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    """Carga audio, lo pasa a mono y lo remuestrea si hace falta."""
    waveform_np, source_rate = sf.read(path, always_2d=True)
    waveform = torch.from_numpy(np.asarray(waveform_np, dtype=np.float32)).transpose(0, 1)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    peak = waveform.abs().max().clamp_min(1e-6)
    return waveform / peak


def save_audio(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Guarda audio en WAV para mantener calidad durante el pipeline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform.squeeze(0).cpu().numpy(), sample_rate)


def resolve_audio_path(candidates: Iterable[Path]) -> Path:
    """Devuelve la primera ruta existente de audio crudo."""
    for path in candidates:
        if path.exists():
            return path
    readable = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No existe ninguno de los audios esperados: {readable}")


def trim_silence(waveform: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    """Recorta silencios al inicio y al final para estabilizar el segmentado."""
    mono = waveform.squeeze(0)
    voiced = torch.where(mono.abs() >= threshold)[0]
    if voiced.numel() == 0:
        return waveform
    start = int(voiced[0].item())
    end = int(voiced[-1].item()) + 1
    return mono[start:end].unsqueeze(0)


def normalize_for_training(waveform: torch.Tensor) -> torch.Tensor:
    """Aplica una normalizacion estable y evita clipping duro."""
    waveform = waveform - waveform.mean(dim=1, keepdim=True)
    peak = waveform.abs().max().clamp_min(1e-6)
    normalized = waveform / peak
    return normalized.clamp(-0.98, 0.98)


def convert_audio_to_wav(source_path: Path, target_path: Path, sample_rate: int) -> Path:
    """Convierte cualquier audio soportado a WAV mono listo para entrenamiento."""
    waveform = load_audio(source_path, sample_rate)
    waveform = trim_silence(normalize_for_training(waveform))
    save_audio(target_path, waveform, sample_rate)
    return target_path


def split_on_silence(
    waveform: torch.Tensor,
    sample_rate: int,
    min_silence_ms: int = 250,
    silence_threshold: float = 0.015,
    min_clip_ms: int = 600,
) -> list[torch.Tensor]:
    """Divide un audio largo usando energia simple por ventanas."""
    mono = waveform.squeeze(0)
    frame_size = max(1, sample_rate // 100)
    if mono.numel() < frame_size * 2:
        return [waveform]
    silence_frames = max(1, int(min_silence_ms / 10))
    min_clip_frames = max(1, int(min_clip_ms / 10))
    rms = mono.unfold(0, frame_size, frame_size).pow(2).mean(dim=1).sqrt()
    voiced = rms > silence_threshold

    clips: list[torch.Tensor] = []
    start = 0
    silent_count = 0
    for index, is_voiced in enumerate(voiced.tolist()):
        if is_voiced:
            silent_count = 0
            continue
        silent_count += 1
        if silent_count < silence_frames:
            continue
        end = max(start + 1, index - silence_frames // 2)
        if end - start >= min_clip_frames:
            clips.append(mono[start * frame_size : end * frame_size].unsqueeze(0))
        start = index + 1
        silent_count = 0

    last_end = voiced.numel()
    if last_end - start >= min_clip_frames:
        clips.append(mono[start * frame_size :].unsqueeze(0))
    return [clip for clip in clips if clip.numel() > frame_size]


def split_evenly_by_count(waveform: torch.Tensor, expected_count: int) -> list[torch.Tensor]:
    """Parte el audio en trozos casi iguales si la deteccion por silencio falla."""
    if expected_count <= 0:
        raise ValueError("La cantidad esperada de segmentos debe ser mayor que cero.")
    if expected_count <= 1:
        return [waveform]
    mono = waveform.squeeze(0)
    total = mono.numel()
    chunk = max(1, total // expected_count)
    clips: list[torch.Tensor] = []
    for index in range(expected_count):
        start = index * chunk
        end = total if index == expected_count - 1 else min(total, (index + 1) * chunk)
        clips.append(mono[start:end].unsqueeze(0))
    return clips


def segment_audio_for_transcript(
    waveform: torch.Tensor,
    sample_rate: int,
    expected_count: int,
    min_silence_ms: int,
    silence_threshold: float,
    min_clip_ms: int,
) -> tuple[list[torch.Tensor], str]:
    """Segmenta el audio y aplica una estrategia de respaldo si es necesario."""
    primary = split_on_silence(
        waveform,
        sample_rate,
        min_silence_ms=min_silence_ms,
        silence_threshold=silence_threshold,
        min_clip_ms=min_clip_ms,
    )
    if len(primary) == expected_count:
        return primary, "silence_split"
    fallback = split_evenly_by_count(waveform, expected_count)
    return fallback, "even_split_fallback"


def build_mel_transform(config: dict) -> torchaudio.transforms.MelSpectrogram:
    """Crea la transformacion a mel espectrograma."""
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=config["sample_rate"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        win_length=config["win_length"],
        n_mels=config["n_mels"],
        power=2.0,
    )


def waveform_to_mel(waveform: torch.Tensor, config: dict) -> torch.Tensor:
    """Convierte un waveform a mel en escala logaritmica."""
    mel = build_mel_transform(config)(waveform)
    return torch.log(mel.clamp_min(1e-5)).squeeze(0).transpose(0, 1)


def mel_to_waveform(mel: torch.Tensor, config: dict) -> torch.Tensor:
    """Reconstruye audio desde mel usando inversion clasica."""
    mel = mel.transpose(0, 1).exp()
    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=config["n_fft"] // 2 + 1,
        n_mels=config["n_mels"],
        sample_rate=config["sample_rate"],
    )
    spectrogram = inverse_mel(mel)
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        win_length=config["win_length"],
        power=2.0,
    )
    return griffin_lim(spectrogram).unsqueeze(0)
