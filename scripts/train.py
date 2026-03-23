"""Entrena el modelo base de TTS."""

from __future__ import annotations

from tts_project.config import MODELS_DIR, PROCESSED_DIR, load_config
from tts_project.trainer import TTSTrainer


def main() -> None:
    """Ejecuta el ciclo de entrenamiento completo."""
    config = load_config()
    manifest_path = PROCESSED_DIR / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError("Primero ejecuta scripts/prepare_dataset.py")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / "tts_model.pt"
    trainer = TTSTrainer(config, manifest_path, checkpoint_path)
    trainer.run()
    print(f"Modelo guardado en: {checkpoint_path}")


if __name__ == "__main__":
    main()
