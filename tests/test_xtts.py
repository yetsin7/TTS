"""Pruebas basicas para la capa XTTS."""

from tts_project.xtts import load_xtts_config


def test_xtts_config_has_language() -> None:
    """Verifica que exista idioma configurado para XTTS."""
    config = load_xtts_config()
    assert config["language"] == "es"
