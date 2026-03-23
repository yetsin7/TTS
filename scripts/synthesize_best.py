"""Usa XTTS si hay fine-tuning; si no, cae al modelo base experimental."""

from __future__ import annotations

import argparse
import subprocess
import sys

from tts_project.config import ROOT_DIR
from tts_project.xtts import load_xtts_config


def xtts_available() -> bool:
    """Determina si existe un checkpoint XTTS fine-tuned listo para inferencia."""
    config = load_xtts_config()
    run_dir = ROOT_DIR / config["output_path"] / config["run_name"]
    return (run_dir / "best_model.pth").exists() and (run_dir / "config.json").exists()


def main() -> None:
    """Despacha la inferencia al backend mas fuerte disponible."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    script = "scripts/infer_xtts.py" if xtts_available() else "scripts/synthesize.py"
    command = [sys.executable, script, "--text", args.text]
    subprocess.run(command, cwd=ROOT_DIR, check=True)


if __name__ == "__main__":
    main()
