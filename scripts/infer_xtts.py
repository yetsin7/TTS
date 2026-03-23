"""Inferencia con XTTS fine-tuned o checkpoint base."""

from __future__ import annotations

import argparse
from pathlib import Path

from tts_project.config import ROOT_DIR
from tts_project.xtts import load_xtts_config


def main() -> None:
    """Genera audio usando XTTS y un speaker reference."""
    try:
        from TTS.api import TTS
    except ImportError as error:  # noqa: BLE001
        raise ImportError("Instala TTS en un entorno compatible antes de usar infer_xtts.py.") from error

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default=str(ROOT_DIR / "outputs" / "generated" / "xtts_output.wav"))
    args = parser.parse_args()

    config = load_xtts_config()
    speaker_wav = str(ROOT_DIR / config["speaker_reference"])
    checkpoint_dir = ROOT_DIR / config["output_path"]
    model_dir = checkpoint_dir / config["run_name"]

    if model_dir.exists():
        tts = TTS(model_path=str(model_dir / "best_model.pth"), config_path=str(model_dir / "config.json"))
    else:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tts.tts_to_file(text=args.text, speaker_wav=speaker_wav, language=config["language"], file_path=str(output_path))
    print(f"Audio XTTS generado en: {output_path}")


if __name__ == "__main__":
    main()
