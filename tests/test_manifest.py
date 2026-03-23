"""Pruebas para validacion del manifiesto."""

from __future__ import annotations

import json
from pathlib import Path

from tts_project.manifest import read_manifest, validate_manifest


def test_manifest_detects_missing_audio(tmp_path: Path) -> None:
    """Verifica que la validacion reporte audios inexistentes."""
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"audio_path": str(tmp_path / "missing.wav"), "text": "hola"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    errors = validate_manifest(manifest)
    assert errors


def test_manifest_reads_rows(tmp_path: Path) -> None:
    """Verifica que el lector del manifiesto preserve las filas."""
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"audio_path": "clip.wav", "text": "hola"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = read_manifest(manifest)
    assert len(rows) == 1
    assert rows[0]["text"] == "hola"
