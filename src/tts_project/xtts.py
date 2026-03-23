"""Utilidades para preparar y ejecutar fine-tuning de XTTS."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from tts_project.config import CONFIG_PATH, PROCESSED_DIR, ROOT_DIR


XTTS_CONFIG_PATH = ROOT_DIR / "configs" / "xtts_finetune.yaml"


def load_xtts_config(path: Path | None = None) -> dict[str, Any]:
    """Carga la configuracion de fine-tuning de XTTS."""
    config_path = path or XTTS_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_manifest(path: Path | None = None) -> list[dict[str, str]]:
    """Lee el manifiesto procesado del proyecto actual."""
    manifest_path = path or (PROCESSED_DIR / "manifest.jsonl")
    return [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def export_xtts_dataset() -> Path:
    """Convierte el dataset procesado actual al formato tipo LJSpeech para XTTS."""
    config = load_xtts_config()
    dataset_dir = ROOT_DIR / config["dataset_path"]
    wavs_dir = dataset_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[str] = []
    for item in read_manifest():
        source = Path(item["audio_path"])
        destination = wavs_dir / source.name
        shutil.copy2(source, destination)
        text = item["text"].replace("|", " ").strip()
        metadata_rows.append(f"{destination.stem}|{text}|{text}")

    metadata_path = dataset_dir / config["metadata_file"]
    metadata_path.write_text("\n".join(metadata_rows) + "\n", encoding="utf-8")
    return metadata_path


def build_dataset_summary() -> dict[str, Any]:
    """Genera un resumen corto del dataset exportado para XTTS."""
    config = load_xtts_config()
    dataset_dir = ROOT_DIR / config["dataset_path"]
    metadata_path = dataset_dir / config["metadata_file"]
    rows = metadata_path.read_text(encoding="utf-8").splitlines() if metadata_path.exists() else []
    return {
        "dataset_dir": str(dataset_dir),
        "metadata_path": str(metadata_path),
        "utterances": len([row for row in rows if row.strip()]),
        "speaker_reference": str(ROOT_DIR / config["speaker_reference"]),
    }
