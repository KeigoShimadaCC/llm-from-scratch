from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from kgpt.byte_bpe import ByteBPETokenizer, train_byte_bpe
from kgpt.config import file_sha256

TOKEN_DATA_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TextRecord:
    record_id: str
    language: str
    text: str


@dataclass(frozen=True)
class PreprocessedRecord:
    record_id: str
    language: str
    text: str
    text_sha256: str
    utf8_bytes: int


@dataclass(frozen=True)
class DedupResult:
    records: tuple[PreprocessedRecord, ...]
    duplicate_record_ids: tuple[str, ...]


@dataclass(frozen=True)
class Batch:
    inputs: torch.Tensor
    targets: torch.Tensor


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return raw


def records_from_config(raw: dict[str, Any]) -> list[TextRecord]:
    dataset = _required_mapping(raw, "dataset")
    source_config = dataset.get("source_config")
    if source_config and "records" not in dataset:
        source_raw = load_yaml_config(str(source_config))
        return records_from_config(source_raw)

    processed_path = dataset.get("processed_path")
    if processed_path:
        return records_from_processed_path(str(processed_path))

    records_raw = dataset.get("records")
    if not isinstance(records_raw, list) or not records_raw:
        raise ValueError("dataset.records must be a non-empty list.")

    records: list[TextRecord] = []
    seen_ids: set[str] = set()
    for item in records_raw:
        if not isinstance(item, dict):
            raise ValueError("dataset.records entries must be mappings.")
        record_id = _required_str(item, "id")
        language = _required_str(item, "language")
        text = _required_str(item, "text")
        if record_id in seen_ids:
            raise ValueError(f"Duplicate dataset record id: {record_id}")
        seen_ids.add(record_id)
        records.append(TextRecord(record_id=record_id, language=language, text=text))
    return records


def records_from_processed_path(processed_path: str | Path) -> list[TextRecord]:
    documents_path = _documents_path(processed_path)
    records: list[TextRecord] = []
    with documents_path.open("r", encoding="utf8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            try:
                record_id = _required_str(item, "doc_id")
                language = _required_str(item, "lang")
                text = _required_str(item, "text")
            except ValueError as exc:
                raise ValueError(f"Invalid processed record at line {line_number}: {exc}") from exc
            records.append(TextRecord(record_id=record_id, language=language, text=text))
    if not records:
        raise ValueError(f"No processed records found: {documents_path}")
    return records


def normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = unicodedata.normalize("NFKC", normalized)
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    return normalized.strip()


def preprocess_and_deduplicate(records: list[TextRecord] | tuple[TextRecord, ...]) -> DedupResult:
    seen_hashes: set[str] = set()
    deduplicated: list[PreprocessedRecord] = []
    duplicates: list[str] = []
    for record in records:
        text = normalize_text(record.text)
        if not text:
            raise ValueError(f"Record {record.record_id} is empty after preprocessing.")
        text_hash = sha256_text(text)
        if text_hash in seen_hashes:
            duplicates.append(record.record_id)
            continue
        seen_hashes.add(text_hash)
        deduplicated.append(
            PreprocessedRecord(
                record_id=record.record_id,
                language=record.language,
                text=text,
                text_sha256=text_hash,
                utf8_bytes=len(text.encode("utf8")),
            )
        )
    if len(deduplicated) < 2:
        raise ValueError("At least two unique records are required for train/validation splitting.")
    return DedupResult(records=tuple(deduplicated), duplicate_record_ids=tuple(duplicates))


def split_records(
    records: list[PreprocessedRecord] | tuple[PreprocessedRecord, ...],
    *,
    validation_fraction: float,
    seed: int,
) -> dict[str, tuple[PreprocessedRecord, ...]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if len(records) < 2:
        raise ValueError("At least two records are required for train/validation splitting.")

    ordered = sorted(
        records,
        key=lambda record: hashlib.sha256(f"{seed}:{record.record_id}:{record.text_sha256}".encode()).hexdigest(),
    )
    validation_count = int(round(len(ordered) * validation_fraction))
    validation_count = min(max(validation_count, 1), len(ordered) - 1)
    validation = tuple(ordered[:validation_count])
    train = tuple(ordered[validation_count:])
    return {"train": train, "validation": validation}


def build_tokenized_dataset_from_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    raw = load_yaml_config(config_path)
    tokenizer = _load_or_train_tokenizer(raw)
    dataset = _required_mapping(raw, "dataset")
    split_manifest_path = dataset.get("split_manifest_path")
    if split_manifest_path:
        splits, dedup = splits_from_manifest_config(dataset)
    else:
        records = records_from_config(raw)
        dedup = preprocess_and_deduplicate(records)
        split_seed = _required_int(dataset, "split_seed")
        validation_fraction = _required_float(dataset, "validation_fraction")
        splits = split_records(dedup.records, validation_fraction=validation_fraction, seed=split_seed)

    output_dir = Path(_required_str(dataset, "output_dir"))
    metadata_path = Path(dataset.get("metadata_path", output_dir / "metadata.json"))
    manifest_path = Path(_required_str(dataset, "manifest_path"))
    metadata = write_tokenized_splits(
        tokenizer=tokenizer,
        splits=splits,
        output_dir=output_dir,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        dataset_config=raw,
        source_config_path=config_path,
        dedup=dedup,
    )
    return metadata


def splits_from_manifest_config(
    dataset: dict[str, Any],
) -> tuple[dict[str, tuple[PreprocessedRecord, ...]], DedupResult]:
    split_manifest_path = Path(_required_str(dataset, "split_manifest_path"))
    processed_path = _required_str(dataset, "processed_path")
    manifest = json.loads(split_manifest_path.read_text(encoding="utf8"))
    processed_records = {
        record.record_id: record
        for record in preprocess_and_deduplicate(records_from_processed_path(processed_path)).records
    }
    splits_payload = manifest.get("records")
    if not isinstance(splits_payload, dict) or not splits_payload:
        raise ValueError("split manifest records must be a non-empty mapping.")

    splits: dict[str, tuple[PreprocessedRecord, ...]] = {}
    seen_record_ids: set[str] = set()
    for split_name, records_raw in splits_payload.items():
        if not isinstance(split_name, str) or not isinstance(records_raw, list):
            raise ValueError("split manifest records entries must be split-name lists.")
        split_records: list[PreprocessedRecord] = []
        for item in records_raw:
            if not isinstance(item, dict):
                raise ValueError("split manifest record entries must be mappings.")
            doc_id = _required_str(item, "doc_id")
            record = processed_records.get(doc_id)
            if record is None:
                raise ValueError(f"split manifest references unknown doc_id: {doc_id}")
            expected_hash = _required_str(item, "normalized_text_sha256")
            if record.text_sha256 != expected_hash:
                raise ValueError(f"split manifest hash mismatch for doc_id: {doc_id}")
            if doc_id in seen_record_ids:
                raise ValueError(f"split manifest repeats doc_id: {doc_id}")
            seen_record_ids.add(doc_id)
            split_records.append(record)
        splits[split_name] = tuple(split_records)
    if "train" not in splits or "validation" not in splits:
        raise ValueError("split manifest must include train and validation splits.")
    all_records = tuple(record for split_records in splits.values() for record in split_records)
    duplicate_ids = tuple(
        item.get("doc_id", "")
        for item in manifest.get("duplicates_removed", [])
        if item.get("doc_id")
    )
    return splits, DedupResult(records=all_records, duplicate_record_ids=duplicate_ids)


def write_tokenized_splits(
    *,
    tokenizer: ByteBPETokenizer,
    splits: dict[str, tuple[PreprocessedRecord, ...]],
    output_dir: Path,
    metadata_path: Path,
    manifest_path: Path,
    dataset_config: dict[str, Any],
    source_config_path: Path,
    dedup: DedupResult,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = np.uint16 if tokenizer.vocab_size <= np.iinfo(np.uint16).max else np.uint32
    split_metadata: dict[str, Any] = {}
    for split_name, split_records_value in splits.items():
        token_ids: list[int] = []
        for record in split_records_value:
            token_ids.extend(tokenizer.encode(record.text, add_eos=True))
        token_array = np.asarray(token_ids, dtype=dtype)
        token_path = output_dir / f"{split_name}.npy"
        np.save(token_path, token_array)
        split_metadata[split_name] = {
            "path": str(token_path),
            "name": split_name,
            "dtype": np.dtype(dtype).name,
            "shape": [int(token_array.shape[0])],
            "token_count": int(token_array.shape[0]),
            "sha256": file_sha256(token_path),
            "record_count": len(split_records_value),
            "record_ids": [record.record_id for record in split_records_value],
            "text_sha256": [record.text_sha256 for record in split_records_value],
        }

    leakage_overlap = _train_heldout_overlap(split_metadata)
    dataset = _required_mapping(dataset_config, "dataset")
    manifest = _source_manifest(
        dataset_config=dataset_config,
        dedup=dedup,
        split_metadata=split_metadata,
        manifest_path=manifest_path,
    )
    metadata = {
        "schema_version": TOKEN_DATA_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source_config_path": str(source_config_path),
        "source_config_sha256": file_sha256(source_config_path),
        "tokenizer": {
            "tokenizer_id": tokenizer.tokenizer_id,
            "algorithm": "byte_bpe",
            "vocab_size": tokenizer.vocab_size,
            "byte_fallback": True,
            "model_path": str(_tokenizer_model_path(dataset_config)),
        },
        "token_file_format": {
            "format": "numpy .npy",
            "layout": "1-D contiguous token ids",
            "dtype": next(iter(split_metadata.values()))["dtype"],
            "target_semantics": "language-model targets are the same stream shifted by one token",
            "mmap_loading": "np.load(path, mmap_mode='r')",
        },
        "split_method": _required_str(dataset, "split_method"),
        "dedup_strategy": _required_str(dataset, "dedup_strategy"),
        "leakage_check": {
            "method": "exact normalized text sha256 intersection between train and held-out splits",
            "overlap_count": len(leakage_overlap),
            "overlap_sha256": leakage_overlap,
        },
        "splits": split_metadata,
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf8")
    return metadata


class TokenBatchSampler:
    def __init__(
        self,
        *,
        metadata_path: str | Path,
        split: str,
        batch_size: int,
        context_length: int,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if context_length <= 0:
            raise ValueError("context_length must be positive.")
        self.metadata_path = Path(metadata_path)
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf8"))
        splits = self.metadata.get("splits")
        if not isinstance(splits, dict) or split not in splits:
            raise ValueError(f"Unknown tokenized split: {split}")
        self.split = split
        self.batch_size = batch_size
        self.context_length = context_length
        self.tokens = np.load(splits[split]["path"], mmap_mode="r")
        if self.tokens.ndim != 1:
            raise ValueError("Token file must be a 1-D token stream.")
        if int(self.tokens.shape[0]) <= context_length:
            raise ValueError("Token file is too short for the requested context_length.")
        self.rng = np.random.default_rng(seed)

    @property
    def vocab_size(self) -> int:
        return int(self.metadata["tokenizer"]["vocab_size"])

    def next_batch(self) -> Batch:
        max_start = int(self.tokens.shape[0]) - self.context_length - 1
        if max_start < 0:
            raise ValueError("Token file is too short to create shifted targets.")
        starts = self.rng.integers(0, max_start + 1, size=self.batch_size)
        inputs = np.stack([self.tokens[start : start + self.context_length] for start in starts]).astype(np.int64)
        targets = np.stack(
            [self.tokens[start + 1 : start + self.context_length + 1] for start in starts]
        ).astype(np.int64)
        return Batch(inputs=torch.tensor(inputs, dtype=torch.long), targets=torch.tensor(targets, dtype=torch.long))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def _source_manifest(
    *,
    dataset_config: dict[str, Any],
    dedup: DedupResult,
    split_metadata: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    dataset = _required_mapping(dataset_config, "dataset")
    total_size = sum(record.utf8_bytes for record in dedup.records)
    source_checksum = hashlib.sha256()
    for text_hash in sorted(record.text_sha256 for record in dedup.records):
        source_checksum.update(text_hash.encode())
    language_counts: dict[str, int] = {}
    for record in dedup.records:
        language_counts[record.language] = language_counts.get(record.language, 0) + 1
    leakage_overlap = _train_heldout_overlap(split_metadata)
    return {
        "schema_version": TOKEN_DATA_SCHEMA_VERSION,
        "manifest_path": str(manifest_path),
        "source_name": _required_str(dataset, "source_name"),
        "url_or_local_note": _required_str(dataset, "url_or_local_note"),
        "license": _required_str(dataset, "license"),
        "language_mix": _required_str(dataset, "language_mix"),
        "language_record_counts": language_counts,
        "size": {
            "unique_records": len(dedup.records),
            "duplicate_records_removed": len(dedup.duplicate_record_ids),
            "utf8_bytes_after_preprocessing": total_size,
        },
        "checksum": {
            "algorithm": "sha256 over sorted unique normalized text hashes",
            "value": source_checksum.hexdigest(),
        },
        "preprocessing_command": _required_str(dataset, "preprocessing_command"),
        "split_method": _required_str(dataset, "split_method"),
        "dedup_strategy": _required_str(dataset, "dedup_strategy"),
        "contamination_leakage_notes": _required_str(dataset, "contamination_leakage_notes"),
        "leakage_check": {
            "method": "exact normalized text sha256 intersection between train and held-out splits",
            "overlap_count": len(leakage_overlap),
            "overlap_sha256": leakage_overlap,
        },
        "duplicate_record_ids_removed": list(dedup.duplicate_record_ids),
        "splits": {
            name: {
                "record_count": payload["record_count"],
                "record_ids": payload["record_ids"],
                "token_count": payload["token_count"],
                "token_file": payload["path"],
                "token_file_sha256": payload["sha256"],
            }
            for name, payload in split_metadata.items()
        },
        "tokenized_metadata_format": "JSON sidecar with token file paths, dtype, sha256, tokenizer id, and split names",
    }


def _load_or_train_tokenizer(raw: dict[str, Any]) -> ByteBPETokenizer:
    tokenizer_path = _tokenizer_model_path(raw)
    if tokenizer_path.is_file():
        return ByteBPETokenizer.load(tokenizer_path)

    training_config_path = _required_mapping(raw, "tokenizer").get("fallback_training_config")
    training_raw = load_yaml_config(str(training_config_path)) if training_config_path else raw
    tokenizer_raw = _required_mapping(training_raw, "tokenizer")
    selected_vocab_size = _required_int(tokenizer_raw, "selected_target_vocab_size")
    min_pair_frequency = _required_int(tokenizer_raw, "min_pair_frequency")
    tokenizer = train_byte_bpe(
        [record.text for record in records_from_config(training_raw)],
        tokenizer_id=_required_str(tokenizer_raw, "tokenizer_id"),
        target_vocab_size=selected_vocab_size,
        min_pair_frequency=min_pair_frequency,
    )
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(tokenizer_path)
    return tokenizer


def _documents_path(processed_path: str | Path) -> Path:
    path = Path(processed_path)
    return path if path.suffix == ".jsonl" else path / "documents.jsonl"


def _train_heldout_overlap(split_metadata: dict[str, Any]) -> list[str]:
    train_hashes = set(split_metadata.get("train", {}).get("text_sha256", []))
    heldout_hashes: set[str] = set()
    for split_name, payload in split_metadata.items():
        if split_name != "train":
            heldout_hashes.update(payload["text_sha256"])
    return sorted(train_hashes & heldout_hashes)


def _tokenizer_model_path(raw: dict[str, Any]) -> Path:
    tokenizer_raw = _required_mapping(raw, "tokenizer")
    return Path(_required_str(tokenizer_raw, "model_path"))


def _required_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing mapping config field: {key}")
    return value


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing string config field: {key}")
    return value


def _required_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Missing integer config field: {key}")
    return value


def _required_float(raw: dict[str, Any], key: str) -> float:
    value = raw.get(key)
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"Missing finite numeric config field: {key}")
    return float(value)
