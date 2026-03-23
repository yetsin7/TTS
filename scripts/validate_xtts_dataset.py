"""Valida el dataset exportado para XTTS siguiendo reglas practicas."""

from __future__ import annotations

import csv
from pathlib import Path

import soundfile as sf

from tts_project.config import ROOT_DIR
from tts_project.xtts import load_xtts_config


def main() -> None:
    """Valida duraciones, textos y speaker reference del dataset XTTS."""
    config = load_xtts_config()
    dataset_dir = ROOT_DIR / config["dataset_path"]
    metadata_path = dataset_dir / config["metadata_file"]
    speaker_reference = ROOT_DIR / config["speaker_reference"]

    if not metadata_path.exists():
        raise FileNotFoundError(f"No existe metadata XTTS: {metadata_path}")
    if not speaker_reference.exists():
        raise FileNotFoundError(f"No existe speaker reference: {speaker_reference}")

    max_duration = float(config["max_audio_seconds"])
    max_text_length = int(config["max_text_length"])
    total = 0
    warnings: list[str] = []

    with metadata_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="|")
        for index, row in enumerate(reader, start=1):
            if len(row) < 2:
                warnings.append(f"Fila {index}: formato invalido en metadata.")
                continue
            clip_id, text = row[0].strip(), row[1].strip()
            wav_path = dataset_dir / "wavs" / f"{clip_id}.wav"
            if not wav_path.exists():
                warnings.append(f"Fila {index}: falta audio {wav_path.name}")
                continue
            info = sf.info(wav_path)
            duration = info.frames / info.samplerate if info.samplerate else 0.0
            if duration > max_duration:
                warnings.append(f"Fila {index}: {wav_path.name} dura {duration:.2f}s y supera {max_duration:.2f}s")
            if len(text) > max_text_length:
                warnings.append(f"Fila {index}: texto de largo {len(text)} supera {max_text_length}")
            total += 1

    print(f"Clips validados: {total}")
    print(f"Speaker reference: {speaker_reference}")
    if warnings:
        print("Advertencias detectadas:")
        for warning in warnings:
            print(f"- {warning}")
        raise SystemExit(1)
    print("Dataset XTTS validado correctamente.")


if __name__ == "__main__":
    main()
