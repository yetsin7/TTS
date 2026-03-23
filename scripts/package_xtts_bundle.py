"""Empaqueta recursos minimos para llevar XTTS a Colab."""

from __future__ import annotations

import shutil
from pathlib import Path

from tts_project.config import ROOT_DIR


def main() -> None:
    """Crea un zip con scripts, config y dataset XTTS."""
    bundle_root = ROOT_DIR / "artifacts" / "xtts_bundle"
    bundle_root.mkdir(parents=True, exist_ok=True)
    archive_base = bundle_root / "xtts_colab_bundle"

    included_paths = [
        ROOT_DIR / "configs" / "xtts_finetune.yaml",
        ROOT_DIR / "requirements-xtts-colab.txt",
        ROOT_DIR / "scripts" / "train_xtts.py",
        ROOT_DIR / "scripts" / "infer_xtts.py",
        ROOT_DIR / "data" / "xtts",
    ]

    staging_dir = bundle_root / "staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    for path in included_paths:
        destination = staging_dir / path.relative_to(ROOT_DIR)
        if path.is_dir():
            shutil.copytree(path, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

    zip_path = shutil.make_archive(str(archive_base), "zip", root_dir=staging_dir)
    print(f"Bundle XTTS generado en: {zip_path}")


if __name__ == "__main__":
    main()
