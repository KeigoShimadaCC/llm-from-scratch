from __future__ import annotations

import copy
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml

from corpus.audit_sources import main as audit_main
from corpus.source_registry import APPROVED_SOURCE_IDS, SourceAuditError, audit_config, validate_config

CONFIG_PATH = Path("configs/corpus_v01.yaml")


def _load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf8"))


def test_corpus_config_loads_and_is_allowlisted() -> None:
    result = audit_config(CONFIG_PATH)

    assert set(result.source_ids) == APPROVED_SOURCE_IDS
    assert "tatoeba" in result.deferred_source_ids
    assert "metadata-only; no corpus payloads downloaded" in result.manifest


def test_source_missing_license_provenance_or_attribution_fails() -> None:
    config = _load_config()

    del config["sources"][0]["license"]["terms_note"]
    del config["sources"][1]["provenance"]["terms_url"]
    del config["sources"][2]["license"]["attribution"]["instructions"]

    errors = validate_config(config)
    joined = "\n".join(errors)
    assert "source enwiki.license.terms_note" in joined
    assert "source jawiki.provenance.terms_url" in joined
    assert "source project_gutenberg.license.attribution.instructions" in joined


def test_unapproved_source_is_rejected() -> None:
    config = _load_config()
    extra = copy.deepcopy(config["sources"][0])
    extra["id"] = "common_crawl"
    extra["language"] = "en"
    config["sources"].append(extra)

    with pytest.raises(SourceAuditError, match="common_crawl"):
        _audit_mapping_via_temp(config)


def test_included_source_with_unverified_terms_is_rejected() -> None:
    config = _load_config()
    config["sources"][0]["license"]["verification_status"] = "unverified"

    errors = validate_config(config)
    assert any("verification_status must be verified" in error for error in errors)


def test_blocked_source_cannot_be_phase_10b_eligible() -> None:
    config = _load_config()
    config["sources"][0]["status"] = "blocked"
    config["sources"][0]["eligible_for_phase_10b"] = True
    config["sources"][0]["license"]["verification_status"] = "blocked"

    errors = validate_config(config)
    assert any("blocked but still eligible_for_phase_10b" in error for error in errors)


def test_storage_paths_are_kept_under_ignored_data_roots() -> None:
    config = _load_config()
    config["sources"][0]["storage"]["raw_path"] = "../data/raw/corpus_v01/enwiki"

    errors = validate_config(config)
    assert any("storage.raw_path must be a safe relative path" in error for error in errors)


def test_tatoeba_must_remain_deferred() -> None:
    config = _load_config()
    config["deferred_sources"] = []

    errors = validate_config(config)
    assert any("Tatoeba" in error for error in errors)


def test_manifest_output_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "manifest_first.md"
    second = tmp_path / "manifest_second.md"

    assert audit_main(["--config", str(CONFIG_PATH), "--output", str(first)]) == 0
    assert audit_main(["--config", str(CONFIG_PATH), "--output", str(second)]) == 0

    assert first.read_text(encoding="utf8") == second.read_text(encoding="utf8")
    assert "## Approved Sources" in first.read_text(encoding="utf8")
    assert "Tatoeba" in first.read_text(encoding="utf8")


def _audit_mapping_via_temp(config: dict) -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "corpus_v01.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf8")
        audit_config(path)
