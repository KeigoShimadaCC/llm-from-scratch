from __future__ import annotations

import json
from pathlib import Path

import yaml

from corpus.cleaning import write_smoke_corpus
from corpus.splitting import write_split_manifest
from kgpt.token_data import build_tokenized_dataset_from_config, records_from_config
from tokenizer.train_report import generate_tokenizer_report


def test_tokenizer_report_reads_processed_corpus(tmp_path: Path) -> None:
    write_smoke_corpus("configs/corpus_v01.yaml", tmp_path / "processed")
    write_split_manifest(
        config_path="configs/corpus_v01.yaml",
        processed_path=tmp_path / "processed",
        output_path=tmp_path / "dataset_manifest.json",
    )
    config = yaml.safe_load(Path("configs/tokenizer_corpus_v01.yaml").read_text(encoding="utf8"))
    config["dataset"]["processed_path"] = str(tmp_path / "processed")
    config["dataset"]["dataset_manifest_path"] = str(tmp_path / "dataset_manifest.json")
    config["tokenizer"]["model_path"] = str(tmp_path / "tokenizers" / "byte_bpe_4k.json")
    config["tokenizer"]["target_vocab_sizes"] = [2000, 4000]
    config_path = tmp_path / "tokenizer_corpus_v01.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf8")
    output_path = tmp_path / "report.md"

    result = generate_tokenizer_report(config_path=config_path, output_path=output_path)
    report = output_path.read_text(encoding="utf8")

    assert result["selected_tokenizer"].tokenizer_id == "kgpt-corpus-v01-byte-bpe-4k"
    assert "PHASE-10D corpus_v01 Tokenizer Report" in report
    assert "byte_bpe_4k" in report


def test_tokenized_dataset_uses_phase10c_split_manifest(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    split_manifest = tmp_path / "dataset_manifest.json"
    tokenizer_model = tmp_path / "tokenizers" / "byte_bpe_4k.json"
    tokenized_dir = tmp_path / "tokenized"
    tokenized_manifest = tmp_path / "corpus_v01_tokenized_manifest.json"
    write_smoke_corpus("configs/corpus_v01.yaml", processed)
    write_split_manifest(config_path="configs/corpus_v01.yaml", processed_path=processed, output_path=split_manifest)

    tokenizer_config = yaml.safe_load(Path("configs/tokenizer_corpus_v01.yaml").read_text(encoding="utf8"))
    tokenizer_config["dataset"]["processed_path"] = str(processed)
    tokenizer_config["dataset"]["dataset_manifest_path"] = str(split_manifest)
    tokenizer_config["tokenizer"]["model_path"] = str(tokenizer_model)
    tokenizer_config["tokenizer"]["target_vocab_sizes"] = [2000, 4000]
    tokenizer_config_path = tmp_path / "tokenizer_corpus_v01.yaml"
    tokenizer_config_path.write_text(yaml.safe_dump(tokenizer_config, allow_unicode=True), encoding="utf8")

    config = yaml.safe_load(Path("configs/corpus_v01_tokenized.yaml").read_text(encoding="utf8"))
    config["tokenizer"]["model_path"] = str(tokenizer_model)
    config["tokenizer"]["fallback_training_config"] = str(tokenizer_config_path)
    config["dataset"]["source_config"] = str(tokenizer_config_path)
    config["dataset"]["processed_path"] = str(processed)
    config["dataset"]["split_manifest_path"] = str(split_manifest)
    config["dataset"]["output_dir"] = str(tokenized_dir)
    config["dataset"]["metadata_path"] = str(tokenized_dir / "metadata.json")
    config["dataset"]["manifest_path"] = str(tokenized_manifest)
    config_path = tmp_path / "corpus_v01_tokenized.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf8")

    metadata = build_tokenized_dataset_from_config(config_path)
    manifest = json.loads(tokenized_manifest.read_text(encoding="utf8"))

    assert set(metadata["splits"]) == {"train", "validation", "test"}
    assert metadata["leakage_check"]["overlap_count"] == 0
    assert manifest["leakage_check"]["overlap_count"] == 0
    assert Path(metadata["splits"]["train"]["path"]).is_file()
    assert Path(metadata["splits"]["validation"]["path"]).is_file()
    assert Path(metadata["splits"]["test"]["path"]).is_file()
    assert len(records_from_config(tokenizer_config)) == 4
