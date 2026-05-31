from pathlib import Path

import yaml

from tokenizer.train_report import generate_tokenizer_report


def test_tokenizer_report_includes_phase_required_sections(tmp_path) -> None:
    config = yaml.safe_load(Path("configs/tokenizer_bilingual.yaml").read_text(encoding="utf8"))
    config["tokenizer"]["model_path"] = str(tmp_path / "tokenizer.json")
    config["tokenizer"]["target_vocab_sizes"] = [8000]
    config["tokenizer"]["selected_target_vocab_size"] = 8000
    config_path = tmp_path / "tokenizer_bilingual.yaml"
    output_path = tmp_path / "tokenizer_report.md"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf8")

    result = generate_tokenizer_report(config_path=config_path, output_path=output_path)
    report = output_path.read_text(encoding="utf8")

    sample = "英語 and 日本語 🧪"
    assert result["selected_tokenizer"].decode(result["selected_tokenizer"].encode(sample)) == sample
    assert "Vocabulary Sweep" in report
    assert "English/Japanese Tokenization" in report
    assert "Unknown And Byte Fallback Behavior" in report
    assert "Compression And Context-Length Effect" in report
    assert "sentencepiece_unigram" in report
    assert "rejected-for-phase" in report
