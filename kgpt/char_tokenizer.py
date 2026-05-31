from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOKENIZER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CharacterTokenizer:
    tokenizer_id: str
    itos: tuple[str, ...]

    @classmethod
    def build(cls, texts: list[str] | tuple[str, ...], *, tokenizer_id: str = "char-v1") -> CharacterTokenizer:
        if not texts:
            raise ValueError("At least one text is required to build a character tokenizer.")
        vocabulary = sorted({character for text in texts for character in text})
        if not vocabulary:
            raise ValueError("Cannot build a character tokenizer from empty text.")
        return cls(tokenizer_id=tokenizer_id, itos=tuple(vocabulary))

    @property
    def stoi(self) -> dict[str, int]:
        return {character: index for index, character in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        stoi = self.stoi
        unknown = sorted({character for character in text if character not in stoi})
        if unknown:
            printable = ", ".join(repr(character) for character in unknown)
            raise ValueError(f"Text contains characters outside the vocabulary: {printable}")
        return [stoi[character] for character in text]

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        characters: list[str] = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"Token id {token_id} is outside vocabulary size {self.vocab_size}.")
            characters.append(self.itos[token_id])
        return "".join(characters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TOKENIZER_SCHEMA_VERSION,
            "tokenizer_id": self.tokenizer_id,
            "vocabulary": list(self.itos),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CharacterTokenizer:
        if payload.get("schema_version") != TOKENIZER_SCHEMA_VERSION:
            raise ValueError("Unsupported character tokenizer schema version.")
        tokenizer_id = payload.get("tokenizer_id")
        vocabulary = payload.get("vocabulary")
        if not isinstance(tokenizer_id, str) or not tokenizer_id:
            raise ValueError("Tokenizer payload is missing tokenizer_id.")
        if not isinstance(vocabulary, list) or not all(isinstance(item, str) and len(item) == 1 for item in vocabulary):
            raise ValueError("Tokenizer payload is missing a single-character vocabulary.")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("Tokenizer vocabulary contains duplicate characters.")
        return cls(tokenizer_id=tokenizer_id, itos=tuple(vocabulary))

    def save(self, path: str | Path) -> None:
        tokenizer_path = Path(path)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf8")

    @classmethod
    def load(cls, path: str | Path) -> CharacterTokenizer:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf8")))
