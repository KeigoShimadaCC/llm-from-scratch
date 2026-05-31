import json
from pathlib import Path

import torch.nn as nn
import yaml

from kgpt.token_data import TokenBatchSampler, build_tokenized_dataset_from_config


def test_tokenized_dataset_dedup_split_leakage_and_batch_shapes(tmp_path) -> None:
    tokenizer_config = yaml.safe_load(Path("configs/tokenizer_bilingual.yaml").read_text(encoding="utf8"))
    tokenizer_config["tokenizer"]["model_path"] = str(tmp_path / "tokenizers" / "byte_bpe.json")
    tokenizer_config_path = tmp_path / "tokenizer_bilingual.yaml"
    tokenizer_config_path.write_text(yaml.safe_dump(tokenizer_config, allow_unicode=True), encoding="utf8")

    config = yaml.safe_load(Path("configs/tokenized_smoke.yaml").read_text(encoding="utf8"))
    config["tokenizer"]["model_path"] = tokenizer_config["tokenizer"]["model_path"]
    config["tokenizer"]["fallback_training_config"] = str(tokenizer_config_path)
    config["dataset"]["source_config"] = str(tokenizer_config_path)
    config["dataset"]["output_dir"] = str(tmp_path / "tokenized")
    config["dataset"]["metadata_path"] = str(tmp_path / "tokenized" / "metadata.json")
    config["dataset"]["manifest_path"] = str(tmp_path / "phase02a_data_manifest.json")
    config_path = tmp_path / "tokenized_smoke.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf8")

    metadata = build_tokenized_dataset_from_config(config_path)
    manifest = json.loads(Path(config["dataset"]["manifest_path"]).read_text(encoding="utf8"))

    assert metadata["token_file_format"]["format"] == "numpy .npy"
    assert metadata["leakage_check"]["overlap_count"] == 0
    assert manifest["size"]["duplicate_records_removed"] == 1
    assert manifest["leakage_check"]["overlap_count"] == 0
    assert set(metadata["splits"]) == {"train", "validation"}
    assert Path(metadata["splits"]["train"]["path"]).is_file()
    assert Path(metadata["splits"]["validation"]["path"]).is_file()

    sampler = TokenBatchSampler(
        metadata_path=config["dataset"]["metadata_path"],
        split="train",
        batch_size=4,
        context_length=16,
        seed=123,
    )
    batch = sampler.next_batch()
    assert list(batch.inputs.shape) == [4, 16]
    assert list(batch.targets.shape) == [4, 16]

    model = nn.Sequential(nn.Embedding(sampler.vocab_size, 8), nn.Linear(8, sampler.vocab_size))
    logits = model(batch.inputs)
    assert list(logits.shape) == [4, 16, sampler.vocab_size]
