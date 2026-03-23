"""Genera audio nuevo a partir de texto usando el modelo entrenado."""

from __future__ import annotations

import argparse
from pathlib import Path

from tts_project.config import OUTPUTS_DIR
from tts_project.service import TTSService


def main() -> None:
    """Carga el checkpoint y sintetiza un WAV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Texto a sintetizar.")
    parser.add_argument("--output", default=str(OUTPUTS_DIR / "generated" / "tts_output.wav"))
    args = parser.parse_args()

    service = TTSService()
    service.synthesize_to_file(args.text, Path(args.output))
    print(f"Audio generado en: {args.output}")


if __name__ == "__main__":
    main()
