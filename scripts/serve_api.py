"""Sirve el modelo TTS por HTTP usando FastAPI."""

from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from tts_project.service import TTSService


class SynthesisRequest(BaseModel):
    """Esquema del cuerpo para sintesis."""

    text: str = Field(..., min_length=1, max_length=1000, description="Texto a sintetizar.")


app = FastAPI(title="TTS API", version="0.1.0")
service = TTSService()


@app.get("/health")
def health() -> dict[str, str]:
    """Permite verificar si la API esta viva."""
    return {"status": "ok"}


@app.post("/synthesize")
def synthesize(request: SynthesisRequest) -> FileResponse:
    """Genera un archivo WAV y lo devuelve al cliente."""
    output_path = service.synthesize_to_file(request.text, Path("outputs/generated/api_tts_output.wav"))
    return FileResponse(path=output_path, media_type="audio/wav", filename=output_path.name)


def main() -> None:
    """Arranca el servidor local."""
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
