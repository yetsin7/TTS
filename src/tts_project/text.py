"""Procesamiento de texto para el modelo TTS."""

from __future__ import annotations

from dataclasses import dataclass

PAD = "_"
UNK = "?"


@dataclass
class TextTokenizer:
    """Tokenizador simple basado en caracteres para arrancar rapido."""

    symbols: str = PAD + UNK + " abcdefghijklmnopqrstuvwxyz0123456789.,;:!?¡¿'-\"()"

    def __post_init__(self) -> None:
        self.stoi = {symbol: index for index, symbol in enumerate(self.symbols)}
        self.itos = {index: symbol for symbol, index in self.stoi.items()}

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def vocab_size(self) -> int:
        return len(self.symbols)

    def normalize(self, text: str) -> str:
        """Normaliza texto para hacerlo mas consistente."""
        return " ".join(text.lower().strip().split())

    def encode(self, text: str) -> list[int]:
        """Convierte texto a ids."""
        normalized = self.normalize(text)
        unknown_id = self.stoi[UNK]
        return [self.stoi.get(char, unknown_id) for char in normalized]

    def decode(self, token_ids: list[int]) -> str:
        """Convierte ids de vuelta a texto."""
        return "".join(self.itos.get(token_id, UNK) for token_id in token_ids)
