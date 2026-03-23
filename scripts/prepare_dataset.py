"""Prepara segmentos de audio y un manifiesto para entrenamiento."""

from __future__ import annotations

import json
from pathlib import Path

from tts_project.audio import convert_audio_to_wav, load_audio, resolve_audio_path, save_audio, segment_audio_for_transcript
from tts_project.config import PROCESSED_DIR, RAW_DIR, load_config


def read_lines(path: Path) -> list[str]:
    """Carga frases no vacias del archivo de texto."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    """Procesa el audio crudo y genera el manifiesto JSONL."""
    config = load_config()
    wav_audio = RAW_DIR / "audio_entrenamiento.wav"
    mp3_audio = RAW_DIR / "audio_entrenamiento.mp3"
    flac_audio = RAW_DIR / "audio_entrenamiento.flac"
    raw_text = RAW_DIR / "texto_entrenamiento.txt"
    raw_audio = resolve_audio_path([wav_audio, mp3_audio, flac_audio])

    if not raw_text.exists():
        raise FileNotFoundError(f"No existe el archivo esperado: {raw_text}")

    lines = read_lines(raw_text)
    if not lines:
        raise ValueError("El archivo de texto esta vacio. Agrega una frase por linea.")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    normalized_audio = PROCESSED_DIR / "audio_normalizado.wav"
    convert_audio_to_wav(raw_audio, normalized_audio, config["sample_rate"])
    waveform = load_audio(normalized_audio, config["sample_rate"])
    clips, strategy = segment_audio_for_transcript(
        waveform,
        config["sample_rate"],
        expected_count=len(lines),
        min_silence_ms=config["min_silence_ms"],
        silence_threshold=config["silence_threshold"],
        min_clip_ms=config["min_clip_ms"],
    )

    manifest_path = PROCESSED_DIR / "manifest.jsonl"
    metadata_path = PROCESSED_DIR / "dataset_metadata.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for index, (text, clip) in enumerate(zip(lines, clips, strict=True)):
            audio_path = PROCESSED_DIR / f"clip_{index:04d}.wav"
            save_audio(audio_path, clip, config["sample_rate"])
            item = {"audio_path": str(audio_path), "text": text}
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    metadata = {
        "source_audio": str(raw_audio),
        "normalized_audio": str(normalized_audio),
        "transcript_lines": len(lines),
        "generated_clips": len(clips),
        "segmentation_strategy": strategy,
        "sample_rate": config["sample_rate"],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Dataset preparado en: {manifest_path}")
    print(f"Muestras generadas: {len(clips)}")
    print(f"Estrategia de segmentado: {strategy}")


if __name__ == "__main__":
    main()
