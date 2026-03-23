"""Verifica si los archivos crudos estan listos para entrenar."""

from __future__ import annotations

from tts_project.audio import resolve_audio_path
from tts_project.raw_data import get_audio_candidates, get_text_path, inspect_audio, read_transcript_lines, validate_transcript


def main() -> None:
    """Imprime rutas exactas, formato recomendado y estado actual."""
    audio_candidates = get_audio_candidates()
    text_path = get_text_path()
    print("Rutas esperadas para entrenamiento:")
    for path in audio_candidates:
        print(f"- Audio permitido: {path}")
    print(f"- Texto requerido: {text_path}")
    print("Formato recomendado: WAV mono o estereo, 16-bit, voz limpia, una frase por linea.")

    try:
        audio_path = resolve_audio_path(audio_candidates)
    except FileNotFoundError as error:
        print(error)
        raise SystemExit(1) from error

    if not text_path.exists():
        print(f"No existe el transcript: {text_path}")
        raise SystemExit(1)

    lines = read_transcript_lines(text_path)
    issues = validate_transcript(lines)
    if issues:
        print("Problemas detectados en el texto:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    audio_info = inspect_audio(audio_path)
    print(f"Audio detectado: {audio_path}")
    print(f"Lineas de transcript: {len(lines)}")
    print(f"Duracion aproximada (s): {audio_info['duration_seconds']:.2f}")
    print(f"Canales: {int(audio_info['channels'])}")
    print(f"Frecuencia de muestreo origen: {int(audio_info['sample_rate'])}")
    print("Preflight correcto. Puedes ejecutar: python scripts/run_pipeline.py")


if __name__ == "__main__":
    main()
