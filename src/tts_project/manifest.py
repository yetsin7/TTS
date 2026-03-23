"""Validacion y estadisticas del manifiesto del dataset."""

from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Lee un manifiesto JSONL y devuelve sus filas."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_manifest(path: Path) -> list[str]:
    """Valida que cada entrada tenga texto y un audio accesible."""
    errors: list[str] = []
    for index, item in enumerate(read_manifest(path), start=1):
        audio_path = Path(item.get("audio_path", ""))
        text = item.get("text", "").strip()
        if not audio_path.exists():
            errors.append(f"Fila {index}: no existe {audio_path}")
        if not text:
            errors.append(f"Fila {index}: el texto esta vacio")
    return errors


def summarize_manifest(path: Path) -> dict[str, float]:
    """Calcula estadisticas utiles para revisar el corpus."""
    rows = read_manifest(path)
    total_seconds = 0.0
    total_chars = 0
    for item in rows:
        info = sf.info(item["audio_path"])
        total_seconds += info.frames / info.samplerate
        total_chars += len(item["text"])

    utterances = len(rows)
    avg_seconds = total_seconds / utterances if utterances else 0.0
    avg_chars = total_chars / utterances if utterances else 0.0
    return {
        "utterances": float(utterances),
        "total_seconds": total_seconds,
        "avg_seconds": avg_seconds,
        "avg_chars": avg_chars,
    }
