"""Reglas y validaciones para los archivos crudos de entrenamiento."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torchaudio

from tts_project.config import RAW_DIR


AUDIO_FILENAMES = [
    "audio_entrenamiento.wav",
    "audio_entrenamiento.mp3",
    "audio_entrenamiento.flac",
]
TEXT_FILENAME = "texto_entrenamiento.txt"


def get_audio_candidates() -> list[Path]:
    """Devuelve las rutas de audio aceptadas por el proyecto."""
    return [RAW_DIR / filename for filename in AUDIO_FILENAMES]


def get_text_path() -> Path:
    """Devuelve la ruta del transcript principal."""
    return RAW_DIR / TEXT_FILENAME


def read_transcript_lines(path: Path) -> list[str]:
    """Lee el transcript y descarta lineas vacias."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def inspect_audio(path: Path) -> dict[str, float]:
    """Lee metadatos basicos del audio con una ruta compatible entre formatos."""
    try:
        info = sf.info(path)
        duration = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
        return {
            "sample_rate": float(info.samplerate),
            "channels": float(info.channels),
            "frames": float(info.frames),
            "duration_seconds": duration,
        }
    except RuntimeError:
        waveform, sample_rate = torchaudio.load(path)
        frames = waveform.shape[-1]
        channels = waveform.shape[0]
        duration = float(frames) / float(sample_rate) if sample_rate else 0.0
        return {
            "sample_rate": float(sample_rate),
            "channels": float(channels),
            "frames": float(frames),
            "duration_seconds": duration,
        }


def validate_transcript(lines: list[str]) -> list[str]:
    """Valida calidad minima del transcript de entrada."""
    issues: list[str] = []
    if not lines:
        issues.append("El archivo de texto esta vacio.")
        return issues
    if len(lines) < 2:
        issues.append("Necesitas al menos 2 lineas para entrenar.")
    if any(len(line) < 3 for line in lines):
        issues.append("Hay lineas demasiado cortas. Usa frases completas.")
    if any(len(line) > 220 for line in lines):
        issues.append("Hay lineas demasiado largas. Conviene dividir frases extensas.")
    return issues
