from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corpus.source_registry import audit_config, load_corpus_config

CLEANING_VERSION = "corpus_v01_cleaning_v1"
REQUIRED_RECORD_FIELDS = (
    "doc_id",
    "source_id",
    "lang",
    "title",
    "text",
    "license",
    "attribution",
    "source_url",
    "source_record_id",
    "sha256",
    "cleaning_version",
)


@dataclass(frozen=True)
class SmokeDocument:
    source_id: str
    source_record_id: str
    title: str
    raw_text: str
    source_url: str


SMOKE_DOCUMENTS: tuple[SmokeDocument, ...] = (
    SmokeDocument(
        source_id="enwiki",
        source_record_id="smoke-enwiki-001",
        title="Smoke English Wikipedia Article",
        source_url="https://dumps.wikimedia.org/enwiki/smoke",
        raw_text=(
            "'''Language models''' use [[attention (machine learning)|attention]] to predict text. "
            "<ref>Repo-authored smoke reference.</ref> {{short description|example}} "
            "[[Category:Smoke tests]] They are trained on tokens."
        ),
    ),
    SmokeDocument(
        source_id="jawiki",
        source_record_id="smoke-jawiki-001",
        title="スモーク日本語記事",
        source_url="https://dumps.wikimedia.org/jawiki/smoke",
        raw_text=(
            "'''言語モデル'''は[[注意機構|注意]]を使って文章を予測する。"
            "<ref>リポジトリ作成の注。</ref>{{仮リンク|例}}[[Category:テスト]]"
        ),
    ),
    SmokeDocument(
        source_id="project_gutenberg",
        source_record_id="smoke-gutenberg-001",
        title="Repo Authored Gutenberg Smoke",
        source_url="https://www.gutenberg.org/ebooks/smoke",
        raw_text=(
            "*** START OF THE PROJECT GUTENBERG EBOOK REPO SMOKE ***\n"
            "This plain English paragraph is authored for the repository smoke corpus. "
            "It has enough words to exercise cleaning and hashing.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK REPO SMOKE ***\n"
            "Project Gutenberg license boilerplate would normally continue here."
        ),
    ),
    SmokeDocument(
        source_id="aozora_bunko",
        source_record_id="smoke-aozora-001",
        title="リポジトリ作成青空スモーク",
        source_url="https://www.aozora.gr.jp/cards/smoke/cardsmoke.html",
        raw_text=(
            "リポジトリ作成青空スモーク\n"
            "-------------------------------------------------------\n"
            "これは｜人工知能《じんこうちのう》を説明するための短い文章です。"
            "※［＃注記、1-1］余分な注記を取り除きます。\n"
            "底本：リポジトリ作成のスモーク本文\n"
        ),
    ),
)


def clean_text_for_source(source_id: str, raw_text: str) -> str:
    if source_id in {"enwiki", "jawiki"}:
        cleaned = _clean_wikipedia(raw_text)
    elif source_id == "project_gutenberg":
        cleaned = _clean_gutenberg(raw_text)
    elif source_id == "aozora_bunko":
        cleaned = _clean_aozora(raw_text)
    else:
        raise ValueError(f"Unsupported corpus_v01 source for cleaning: {source_id}")
    return normalize_text(cleaned)


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return text.strip()


def build_smoke_records(config_path: str | Path) -> list[dict[str, Any]]:
    audit_config(config_path)
    config = load_corpus_config(config_path)
    sources = {source["id"]: source for source in config["sources"]}
    records: list[dict[str, Any]] = []
    for document in SMOKE_DOCUMENTS:
        source = sources[document.source_id]
        text = clean_text_for_source(document.source_id, document.raw_text)
        if not text:
            raise ValueError(f"Smoke document cleaned to empty text: {document.source_record_id}")
        text_hash = hashlib.sha256(text.encode("utf8")).hexdigest()
        doc_id = hashlib.sha256(
            f"{document.source_id}:{document.source_record_id}:{CLEANING_VERSION}".encode()
        ).hexdigest()[:24]
        records.append(
            {
                "doc_id": doc_id,
                "source_id": document.source_id,
                "lang": source["language"],
                "title": document.title,
                "text": text,
                "license": source["license"]["license_name"],
                "attribution": source["license"]["attribution"]["instructions"],
                "source_url": document.source_url,
                "source_record_id": document.source_record_id,
                "sha256": text_hash,
                "cleaning_version": CLEANING_VERSION,
            }
        )
    return records


def validate_processed_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_RECORD_FIELDS:
        if field not in record:
            errors.append(f"{field} is required")
    for field in REQUIRED_RECORD_FIELDS:
        if field in record and not isinstance(record[field], str):
            errors.append(f"{field} must be a string")
    if record.get("source_id") not in {"enwiki", "jawiki", "project_gutenberg", "aozora_bunko"}:
        errors.append("source_id must be approved for corpus_v01")
    if record.get("lang") not in {"en", "ja"}:
        errors.append("lang must be en or ja")
    text = record.get("text")
    if isinstance(text, str):
        expected = hashlib.sha256(text.encode("utf8")).hexdigest()
        if record.get("sha256") != expected:
            errors.append("sha256 must match normalized text")
    return errors


def write_smoke_corpus(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    records = build_smoke_records(config_path)
    for record in records:
        errors = validate_processed_record(record)
        if errors:
            raise ValueError(f"Invalid processed smoke record {record.get('source_record_id')}: {errors}")

    documents_path = output_path / "documents.jsonl"
    with documents_path.open("w", encoding="utf8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "schema_version": 1,
        "corpus_id": "corpus_v01",
        "mode": "smoke",
        "cleaning_version": CLEANING_VERSION,
        "documents_path": documents_path.as_posix(),
        "document_count": len(records),
        "source_counts": _count_by(records, "source_id"),
        "language_counts": _count_by(records, "lang"),
        "record_sha256": hashlib.sha256(documents_path.read_bytes()).hexdigest(),
        "artifact_policy": "ignored local artifact; do not commit data/processed/**",
    }
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf8")
    return manifest


def _clean_wikipedia(value: str) -> str:
    text = re.sub(r"<ref\b[^>]*>.*?</ref>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"\[\[Category:[^\]]+\]\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[(?:File|Image):[^\]]+\]\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"'{2,}", "", text)
    return text


def _clean_gutenberg(value: str) -> str:
    start = re.search(r"\*\*\*\s*START OF (?:THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*", value, re.IGNORECASE)
    end = re.search(r"\*\*\*\s*END OF (?:THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*", value, re.IGNORECASE)
    if start and end and start.end() < end.start():
        value = value[start.end() : end.start()]
    value = re.sub(r"Project Gutenberg(?:-tm)?", " ", value, flags=re.IGNORECASE)
    return value


def _clean_aozora(value: str) -> str:
    text = re.sub(r"^-{5,}$", " ", value, flags=re.MULTILINE)
    text = re.sub(r"※［＃.*?］", " ", text)
    text = re.sub(r"［＃.*?］", " ", text)
    text = re.sub(r"《.*?》", " ", text)
    text = text.replace("｜", "")
    text = re.sub(r"^底本：.*$", " ", text, flags=re.MULTILINE)
    return text


def _count_by(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(record[field])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
