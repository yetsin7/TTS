"""Prueba simple del tokenizador base."""

from tts_project.text import TextTokenizer


def test_tokenizer_encode_decode() -> None:
    """Verifica que el tokenizador produzca ids estables."""
    tokenizer = TextTokenizer()
    encoded = tokenizer.encode("Hola mundo")
    decoded = tokenizer.decode(encoded)
    assert encoded
    assert "hola" in decoded
