from __future__ import annotations

import json
from pathlib import Path

import yaml

from corpus.clean import main as clean_main
from corpus.cleaning import build_smoke_records, clean_text_for_source, validate_processed_record, write_smoke_corpus
from corpus.download import build_download_plan

CONFIG_PATH = Path("configs/corpus_v01.yaml")


def test_download_dry_run_plan_has_all_allowed_sources() -> None:
    plan = build_download_plan(CONFIG_PATH)

    assert plan["payload_downloads_performed"] is False
    assert plan["source_count"] == 4
    assert plan["blockers"] == []
    assert {entry["source_id"] for entry in plan["sources"]} == {
        "enwiki",
        "jawiki",
        "project_gutenberg",
        "aozora_bunko",
    }
    assert all(entry["action"] == "would_plan_download" for entry in plan["sources"])
    assert all(str(entry["raw_path"]).startswith("data/raw/corpus_v01/") for entry in plan["sources"])


def test_download_plan_rejects_unapproved_source(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf8"))
    config["allowed_source_ids"].append("common_crawl")
    path = tmp_path / "corpus_v01.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf8")

    try:
        build_download_plan(path)
    except Exception as exc:  # noqa: BLE001 - assertion keeps the error text explicit
        assert "common_crawl" in str(exc)
    else:
        raise AssertionError("unapproved source should fail")


def test_cleaning_transforms_remove_source_noise() -> None:
    wiki = clean_text_for_source(
        "enwiki",
        "'''Text''' with [[Link target|visible text]] <ref>noise</ref> {{template}} [[Category:Noise]]",
    )
    gutenberg = clean_text_for_source(
        "project_gutenberg",
        "*** START OF THE PROJECT GUTENBERG EBOOK X ***\nBody text.\n*** END OF THE PROJECT GUTENBERG EBOOK X ***",
    )
    aozora = clean_text_for_source("aozora_bunko", "これは｜単語《たんご》です。※［＃注］\n底本：削除")

    assert wiki == "Text with visible text"
    assert gutenberg == "Body text."
    assert aozora == "これは単語 です。"


def test_smoke_records_have_required_schema_and_hashes() -> None:
    records = build_smoke_records(CONFIG_PATH)

    assert len(records) == 4
    for record in records:
        assert validate_processed_record(record) == []
        assert record["text"]
        assert record["license"]
        assert record["attribution"]


def test_clean_smoke_writes_deterministic_jsonl_and_manifest(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = write_smoke_corpus(CONFIG_PATH, first)
    write_smoke_corpus(CONFIG_PATH, second)

    first_jsonl = (first / "documents.jsonl").read_text(encoding="utf8")
    second_jsonl = (second / "documents.jsonl").read_text(encoding="utf8")
    assert first_jsonl == second_jsonl
    assert first_manifest["document_count"] == 4
    assert first_manifest["source_counts"] == {
        "aozora_bunko": 1,
        "enwiki": 1,
        "jawiki": 1,
        "project_gutenberg": 1,
    }
    assert json.loads((first / "manifest.json").read_text(encoding="utf8"))["record_sha256"]


def test_clean_cli_smoke_output(tmp_path: Path) -> None:
    output = tmp_path / "processed"

    assert clean_main(["--config", str(CONFIG_PATH), "--smoke", "--output", str(output)]) == 0

    assert (output / "documents.jsonl").exists()
    assert (output / "manifest.json").exists()
