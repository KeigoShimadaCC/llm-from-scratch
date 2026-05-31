from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BYTE_BPE_SCHEMA_VERSION = 1
BYTE_BPE_ALGORITHM = "byte_bpe"
SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")

Pair = tuple[bytes, bytes]


@dataclass(frozen=True)
class ByteBPETokenizer:
    tokenizer_id: str
    merges: tuple[Pair, ...]
    special_tokens: tuple[str, ...] = SPECIAL_TOKENS

    def __post_init__(self) -> None:
        if not self.tokenizer_id:
            raise ValueError("tokenizer_id is required.")
        if len(self.special_tokens) != len(set(self.special_tokens)):
            raise ValueError("special_tokens must be unique.")
        merged_tokens = [left + right for left, right in self.merges]
        if len(merged_tokens) != len(set(merged_tokens)):
            raise ValueError("BPE merges produce duplicate vocabulary tokens.")

    @property
    def pad_token_id(self) -> int:
        return self.special_tokens.index("<pad>")

    @property
    def unk_token_id(self) -> int:
        return self.special_tokens.index("<unk>")

    @property
    def bos_token_id(self) -> int:
        return self.special_tokens.index("<bos>")

    @property
    def eos_token_id(self) -> int:
        return self.special_tokens.index("<eos>")

    @property
    def byte_offset(self) -> int:
        return len(self.special_tokens)

    @property
    def merge_offset(self) -> int:
        return self.byte_offset + 256

    @property
    def vocab_size(self) -> int:
        return self.merge_offset + len(self.merges)

    @property
    def merge_count(self) -> int:
        return len(self.merges)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        pieces = self.token_pieces(text)
        token_to_id = self._byte_token_to_id()
        token_ids = [token_to_id[piece] for piece in pieces]
        if add_bos:
            token_ids.insert(0, self.bos_token_id)
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        chunks: list[bytes] = []
        merged_tokens = [left + right for left, right in self.merges]
        for token_id in token_ids:
            if token_id in {self.pad_token_id, self.bos_token_id, self.eos_token_id}:
                continue
            if token_id == self.unk_token_id:
                chunks.append("�".encode())
            elif self.byte_offset <= token_id < self.merge_offset:
                chunks.append(bytes([token_id - self.byte_offset]))
            elif self.merge_offset <= token_id < self.vocab_size:
                chunks.append(merged_tokens[token_id - self.merge_offset])
            else:
                raise ValueError(f"Token id {token_id} is outside vocabulary size {self.vocab_size}.")
        return b"".join(chunks).decode("utf8", errors="replace")

    def token_pieces(self, text: str) -> tuple[bytes, ...]:
        pieces = tuple(bytes([byte]) for byte in text.encode("utf8"))
        for left, right in self.merges:
            pieces = tuple(_merge_pair(list(pieces), (left, right)))
        return pieces

    def count_unknowns(self, token_ids: list[int] | tuple[int, ...]) -> int:
        return sum(1 for token_id in token_ids if token_id == self.unk_token_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BYTE_BPE_SCHEMA_VERSION,
            "algorithm": BYTE_BPE_ALGORITHM,
            "tokenizer_id": self.tokenizer_id,
            "special_tokens": list(self.special_tokens),
            "byte_fallback": True,
            "normalization": "raw utf-8 bytes; no text normalization inside tokenizer",
            "merges": [[left.hex(), right.hex()] for left, right in self.merges],
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ByteBPETokenizer:
        if payload.get("schema_version") != BYTE_BPE_SCHEMA_VERSION:
            raise ValueError("Unsupported byte BPE tokenizer schema version.")
        if payload.get("algorithm") != BYTE_BPE_ALGORITHM:
            raise ValueError("Tokenizer payload is not a byte_bpe tokenizer.")
        tokenizer_id = payload.get("tokenizer_id")
        special_tokens = payload.get("special_tokens")
        merges_raw = payload.get("merges")
        if not isinstance(tokenizer_id, str) or not tokenizer_id:
            raise ValueError("Tokenizer payload is missing tokenizer_id.")
        if not isinstance(special_tokens, list) or not all(isinstance(item, str) for item in special_tokens):
            raise ValueError("Tokenizer payload is missing special_tokens.")
        if not isinstance(merges_raw, list):
            raise ValueError("Tokenizer payload is missing merges.")

        merges: list[Pair] = []
        for item in merges_raw:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not isinstance(item[1], str)
            ):
                raise ValueError("Tokenizer merge entries must be two hex strings.")
            merges.append((bytes.fromhex(item[0]), bytes.fromhex(item[1])))
        return cls(tokenizer_id=tokenizer_id, merges=tuple(merges), special_tokens=tuple(special_tokens))

    def save(self, path: str | Path) -> None:
        tokenizer_path = Path(path)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf8")

    @classmethod
    def load(cls, path: str | Path) -> ByteBPETokenizer:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf8")))

    def _byte_token_to_id(self) -> dict[bytes, int]:
        mapping = {bytes([byte]): self.byte_offset + byte for byte in range(256)}
        for index, (left, right) in enumerate(self.merges):
            mapping[left + right] = self.merge_offset + index
        return mapping


def train_byte_bpe(
    texts: list[str] | tuple[str, ...],
    *,
    tokenizer_id: str,
    target_vocab_size: int,
    min_pair_frequency: int = 2,
    max_merges: int | None = None,
) -> ByteBPETokenizer:
    if not texts:
        raise ValueError("At least one text is required to train byte BPE.")
    if target_vocab_size < len(SPECIAL_TOKENS) + 256:
        raise ValueError("target_vocab_size must leave room for special tokens and all 256 byte tokens.")
    if min_pair_frequency < 1:
        raise ValueError("min_pair_frequency must be positive.")

    sequences = [[bytes([byte]) for byte in text.encode("utf8")] for text in texts]
    if not any(sequences):
        raise ValueError("Cannot train byte BPE from empty text.")

    merges: list[Pair] = []
    vocabulary_tokens = {bytes([byte]) for byte in range(256)}
    merge_budget = target_vocab_size - len(SPECIAL_TOKENS) - 256
    if max_merges is not None:
        merge_budget = min(merge_budget, max_merges)

    while len(merges) < merge_budget:
        pair_counts: Counter[Pair] = Counter()
        for sequence in sequences:
            pair_counts.update(zip(sequence, sequence[1:], strict=False))
        candidates = [
            (pair, count)
            for pair, count in pair_counts.items()
            if count >= min_pair_frequency and pair[0] + pair[1] not in vocabulary_tokens
        ]
        if not candidates:
            break

        best_pair, _count = sorted(
            candidates,
            key=lambda item: (-item[1], item[0][0] + b"\x00" + item[0][1]),
        )[0]
        merged_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocabulary_tokens.add(merged_token)
        sequences = [_merge_pair(sequence, best_pair) for sequence in sequences]

    return ByteBPETokenizer(tokenizer_id=tokenizer_id, merges=tuple(merges))


def display_piece(piece: bytes) -> str:
    try:
        decoded = piece.decode("utf8")
    except UnicodeDecodeError:
        return f"0x{piece.hex()}"
    if decoded == "\n":
        return "\\n"
    if decoded == "\t":
        return "\\t"
    return decoded if decoded.strip() else repr(decoded)


def _merge_pair(sequence: list[bytes], pair: Pair) -> list[bytes]:
    merged: list[bytes] = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and sequence[index] == pair[0] and sequence[index + 1] == pair[1]:
            merged.append(pair[0] + pair[1])
            index += 2
        else:
            merged.append(sequence[index])
            index += 1
    return merged
