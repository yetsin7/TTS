"""Ejecuta el pipeline principal del sistema TTS."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def run_step(command: list[str]) -> None:
    """Ejecuta un paso y detiene el pipeline si falla."""
    print(f"Ejecutando: {' '.join(command)}")
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def main() -> None:
    """Corre preparacion, inspeccion y entrenamiento."""
    python = sys.executable
    run_step([python, "scripts/preflight.py"])
    run_step([python, "scripts/prepare_dataset.py"])
    run_step([python, "scripts/inspect_dataset.py"])
    run_step([python, "scripts/train.py"])


if __name__ == "__main__":
    main()
