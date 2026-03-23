"""Pruebas para validaciones de archivos crudos."""

from __future__ import annotations

from tts_project.raw_data import validate_transcript


def test_validate_transcript_requiere_multiples_lineas() -> None:
    """Verifica que se pidan al menos dos frases."""
    issues = validate_transcript(["Hola"])
    assert issues


def test_validate_transcript_detecta_lineas_largas() -> None:
    """Verifica que se marquen frases demasiado extensas."""
    issues = validate_transcript(["hola mundo", "a" * 221])
    assert issues
