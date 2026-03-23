"""Inspecciona y valida el dataset procesado."""

from __future__ import annotations

from tts_project.config import PROCESSED_DIR
from tts_project.manifest import summarize_manifest, validate_manifest


def main() -> None:
    """Imprime validaciones y estadisticas del manifiesto."""
    manifest_path = PROCESSED_DIR / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError("Primero ejecuta scripts/prepare_dataset.py")

    errors = validate_manifest(manifest_path)
    if errors:
        print("Se encontraron problemas en el manifiesto:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    stats = summarize_manifest(manifest_path)
    print("Dataset validado correctamente.")
    print(f"Utterances: {int(stats['utterances'])}")
    print(f"Duracion total (s): {stats['total_seconds']:.2f}")
    print(f"Duracion promedio (s): {stats['avg_seconds']:.2f}")
    print(f"Caracteres promedio: {stats['avg_chars']:.2f}")


if __name__ == "__main__":
    main()
