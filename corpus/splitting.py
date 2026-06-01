from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from corpus.cleaning import normalize_text, validate_processed_record
from corpus.source_registry import audit_config

DEFAULT_SPLIT_SEED = 1729
DEFAULT_SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


class SplitManifestError(ValueError):
    """Raised when processed corpus splitting or leakage checks fail."""


@dataclass(frozen=True)
class SplitRecord:
    doc_id: str
    source_id: str
    lang: str
    title: str
    source_record_id: str
    text_sha256: str
    normalized_text_sha256: str
    byte_count: int
    char_count: int


def load_processed_records(processed_path: str | Path) -> list[dict[str, Any]]:
    documents_path = _documents_path(processed_path)
    if not documents_path.exists():
        raise SplitManifestError(f"Processed documents not found: {documents_path}")
    records: list[dict[str, Any]] = []
    with documents_path.open("r", encoding="utf8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            errors = validate_processed_record(record)
            if errors:
                raise SplitManifestError(f"Invalid processed record at line {line_number}: {errors}")
            records.append(record)
    if not records:
        raise SplitManifestError(f"No processed records found: {documents_path}")
    return records


def build_split_manifest(
    *,
    config_path: str | Path,
    processed_path: str | Path,
    split_seed: int = DEFAULT_SPLIT_SEED,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    audit = audit_config(config_path)
    raw_records = load_processed_records(processed_path)
    split_records, duplicates = deduplicate_records(raw_records)
    splits = assign_splits(split_records, split_seed=split_seed)
    leakage = check_split_leakage(splits)
    if not leakage["passed"]:
        raise SplitManifestError(f"Split leakage detected: {leakage['errors']}")

    manifest_records = {
        split_name: [record.__dict__ for record in records]
        for split_name, records in splits.items()
    }
    return {
        "schema_version": 1,
        "corpus_id": "corpus_v01",
        "config_path": str(config_path),
        "config_sha256": audit.config_hash,
        "processed_path": str(processed_path),
        "documents_path": _documents_path(processed_path).as_posix(),
        "generated_at_utc": generated_at_utc or datetime.now(UTC).replace(microsecond=0).isoformat(),
        "split_seed": split_seed,
        "split_ratios": DEFAULT_SPLIT_RATIOS,
        "input_document_count": len(raw_records),
        "deduplicated_document_count": len(split_records),
        "duplicate_document_count": len(duplicates),
        "duplicates_removed": duplicates,
        "excluded_count": 0,
        "document_counts": {name: len(records) for name, records in splits.items()},
        "source_counts": _counts_by(split_records, "source_id"),
        "language_counts": _counts_by(split_records, "lang"),
        "split_source_counts": {
            name: _counts_by(records, "source_id")
            for name, records in splits.items()
        },
        "split_language_counts": {
            name: _counts_by(records, "lang")
            for name, records in splits.items()
        },
        "total_bytes": sum(record.byte_count for record in split_records),
        "total_characters": sum(record.char_count for record in split_records),
        "leakage_checks": leakage,
        "records": manifest_records,
        "limitations": [
            "PHASE-10C performs exact normalized-text deduplication only.",
            "Smoke corpus mixture is not representative of the full local corpus.",
            "Committed manifest stores metadata and hashes, not full document text.",
        ],
    }


def write_split_manifest(
    *,
    config_path: str | Path,
    processed_path: str | Path,
    output_path: str | Path,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, Any]:
    manifest = build_split_manifest(config_path=config_path, processed_path=processed_path, split_seed=split_seed)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf8")
    return manifest


def deduplicate_records(records: list[dict[str, Any]]) -> tuple[list[SplitRecord], list[dict[str, str]]]:
    seen_hashes: dict[str, str] = {}
    deduped: list[SplitRecord] = []
    duplicates: list[dict[str, str]] = []
    for record in records:
        normalized = normalize_text(record["text"])
        normalized_hash = hashlib.sha256(normalized.encode("utf8")).hexdigest()
        if normalized_hash in seen_hashes:
            duplicates.append(
                {
                    "doc_id": record["doc_id"],
                    "source_record_id": record["source_record_id"],
                    "duplicate_of_doc_id": seen_hashes[normalized_hash],
                    "normalized_text_sha256": normalized_hash,
                }
            )
            continue
        seen_hashes[normalized_hash] = record["doc_id"]
        deduped.append(
            SplitRecord(
                doc_id=record["doc_id"],
                source_id=record["source_id"],
                lang=record["lang"],
                title=record["title"],
                source_record_id=record["source_record_id"],
                text_sha256=record["sha256"],
                normalized_text_sha256=normalized_hash,
                byte_count=len(record["text"].encode("utf8")),
                char_count=len(record["text"]),
            )
        )
    return deduped, duplicates


def assign_splits(records: list[SplitRecord], *, split_seed: int) -> dict[str, list[SplitRecord]]:
    if not records:
        raise SplitManifestError("Cannot split an empty record set.")
    ordered = sorted(
        records,
        key=lambda record: hashlib.sha256(f"{split_seed}:{record.normalized_text_sha256}".encode()).hexdigest(),
    )
    counts = _split_counts(len(ordered))
    train_end = counts["train"]
    validation_end = train_end + counts["validation"]
    return {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }


def check_split_leakage(splits: dict[str, list[SplitRecord]]) -> dict[str, Any]:
    errors: list[str] = []
    seen_hashes: dict[str, str] = {}
    seen_source_record_ids: dict[str, str] = {}
    for split_name, records in splits.items():
        for record in records:
            previous_split = seen_hashes.get(record.normalized_text_sha256)
            if previous_split and previous_split != split_name:
                errors.append(
                    f"normalized_text_sha256 {record.normalized_text_sha256} appears in "
                    f"{previous_split} and {split_name}"
                )
            seen_hashes.setdefault(record.normalized_text_sha256, split_name)
            previous_source_split = seen_source_record_ids.get(record.source_record_id)
            if previous_source_split and previous_source_split != split_name:
                errors.append(
                    f"source_record_id {record.source_record_id} appears in "
                    f"{previous_source_split} and {split_name}"
                )
            seen_source_record_ids.setdefault(record.source_record_id, split_name)
    return {
        "passed": len(errors) == 0,
        "checks": [
            "normalized_text_sha256_unique_across_splits",
            "source_record_id_unique_across_splits",
        ],
        "errors": errors,
    }


def _documents_path(processed_path: str | Path) -> Path:
    path = Path(processed_path)
    return path if path.suffix == ".jsonl" else path / "documents.jsonl"


def _split_counts(total: int) -> dict[str, int]:
    if total == 1:
        return {"train": 1, "validation": 0, "test": 0}
    if total == 2:
        return {"train": 1, "validation": 1, "test": 0}
    validation = max(1, round(total * DEFAULT_SPLIT_RATIOS["validation"]))
    test = max(1, round(total * DEFAULT_SPLIT_RATIOS["test"]))
    train = total - validation - test
    if train < 1:
        train = 1
        if validation >= test and validation > 0:
            validation -= 1
        elif test > 0:
            test -= 1
    return {"train": train, "validation": validation, "test": test}


def _counts_by(records: list[SplitRecord], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(getattr(record, field))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
