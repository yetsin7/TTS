"""Exporta el dataset local al formato esperado por XTTS."""

from __future__ import annotations

import json

from tts_project.xtts import build_dataset_summary, export_xtts_dataset


def main() -> None:
    """Genera el dataset XTTS y muestra un resumen."""
    metadata_path = export_xtts_dataset()
    summary = build_dataset_summary()
    print(f"Metadata XTTS generada en: {metadata_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
