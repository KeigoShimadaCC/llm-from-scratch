from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpus.cleaning import build_smoke_records, write_smoke_corpus
from corpus.splitting import (
    SplitManifestError,
    assign_splits,
    build_split_manifest,
    check_split_leakage,
    deduplicate_records,
    write_split_manifest,
)

CONFIG_PATH = Path("configs/corpus_v01.yaml")


def test_split_manifest_is_document_level_and_schema_safe(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    output = tmp_path / "manifest.json"
    write_smoke_corpus(CONFIG_PATH, processed)

    manifest = write_split_manifest(config_path=CONFIG_PATH, processed_path=processed, output_path=output)

    assert manifest["schema_version"] == 1
    assert manifest["input_document_count"] == 4
    assert manifest["deduplicated_document_count"] == 4
    assert manifest["document_counts"] == {"test": 1, "train": 2, "validation": 1}
    assert manifest["leakage_checks"]["passed"] is True
    output_text = output.read_text(encoding="utf8")
    assert "This plain English paragraph" not in output_text
    assert "これは単語" not in output_text
    assert "normalized_text_sha256" in output_text


def test_split_assignment_is_deterministic(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    write_smoke_corpus(CONFIG_PATH, processed)

    first = build_split_manifest(config_path=CONFIG_PATH, processed_path=processed, generated_at_utc="fixed")
    second = build_split_manifest(config_path=CONFIG_PATH, processed_path=processed, generated_at_utc="fixed")

    assert first == second


def test_exact_duplicate_text_is_removed_before_split() -> None:
    records = build_smoke_records(CONFIG_PATH)
    duplicate = dict(records[0])
    duplicate["doc_id"] = "duplicate-doc"
    duplicate["source_record_id"] = "duplicate-source-record"

    deduped, duplicates = deduplicate_records([*records, duplicate])

    assert len(deduped) == len(records)
    assert duplicates == [
        {
            "doc_id": "duplicate-doc",
            "source_record_id": "duplicate-source-record",
            "duplicate_of_doc_id": records[0]["doc_id"],
            "normalized_text_sha256": deduped[0].normalized_text_sha256,
        }
    ]


def test_leakage_probe_fails_on_cross_split_source_record_collision() -> None:
    records, _ = deduplicate_records(build_smoke_records(CONFIG_PATH))
    splits = assign_splits(records, split_seed=1729)
    train_record = splits["train"][0]
    leaked = type(train_record)(
        doc_id="leaked-doc",
        source_id=train_record.source_id,
        lang=train_record.lang,
        title=train_record.title,
        source_record_id=train_record.source_record_id,
        text_sha256="different",
        normalized_text_sha256="different-normalized",
        byte_count=10,
        char_count=10,
    )
    splits["validation"].append(leaked)

    leakage = check_split_leakage(splits)

    assert leakage["passed"] is False
    assert any("source_record_id" in error for error in leakage["errors"])


def test_manifest_command_requires_processed_documents(tmp_path: Path) -> None:
    with pytest.raises(SplitManifestError, match="Processed documents not found"):
        build_split_manifest(config_path=CONFIG_PATH, processed_path=tmp_path / "missing")


def test_manifest_json_has_no_full_document_text(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    output = tmp_path / "manifest.json"
    write_smoke_corpus(CONFIG_PATH, processed)
    write_split_manifest(config_path=CONFIG_PATH, processed_path=processed, output_path=output)

    payload = json.loads(output.read_text(encoding="utf8"))

    for split_records in payload["records"].values():
        for record in split_records:
            assert "text" not in record
            assert set(record) >= {"doc_id", "source_id", "lang", "source_record_id", "normalized_text_sha256"}
