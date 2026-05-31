from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from kgpt.token_data import TokenBatchSampler, build_tokenized_dataset_from_config, load_yaml_config


class TinyTokenProbe(nn.Module):
    def __init__(self, *, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.output(self.embedding(token_ids))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample token-level language-model batches from PHASE-02A data.")
    parser.add_argument("--config", required=True, help="Path to tokenized smoke YAML config.")
    parser.add_argument("--max-batches", type=int, default=1, help="Number of batches to sample.")
    args = parser.parse_args(argv)
    if args.max_batches <= 0:
        raise ValueError("--max-batches must be positive.")

    result = sample_batches(config_path=Path(args.config), max_batches=args.max_batches)
    print(json.dumps(result, sort_keys=True))
    return 0


def sample_batches(*, config_path: Path, max_batches: int) -> dict[str, Any]:
    raw = load_yaml_config(config_path)
    metadata = build_tokenized_dataset_from_config(config_path)
    dataset = _required_mapping(raw, "dataset")
    batch_config = _required_mapping(raw, "batch")
    model_config = _required_mapping(batch_config, "small_model")
    sampler = TokenBatchSampler(
        metadata_path=_required_str(dataset, "metadata_path"),
        split=_required_str(batch_config, "split"),
        batch_size=_required_int(batch_config, "batch_size"),
        context_length=_required_int(batch_config, "context_length"),
        seed=_required_int(batch_config, "seed"),
    )
    torch.manual_seed(_required_int(batch_config, "seed"))
    model = TinyTokenProbe(
        vocab_size=sampler.vocab_size,
        embedding_dim=_required_int(model_config, "embedding_dim"),
    )
    criterion = nn.CrossEntropyLoss()
    batches: list[dict[str, Any]] = []
    last_loss = 0.0
    for index in range(max_batches):
        batch = sampler.next_batch()
        logits = model(batch.inputs)
        loss = criterion(logits.reshape(-1, sampler.vocab_size), batch.targets.reshape(-1))
        last_loss = float(loss.detach().cpu())
        batches.append(
            {
                "batch_index": index,
                "input_shape": list(batch.inputs.shape),
                "target_shape": list(batch.targets.shape),
                "logit_shape": list(logits.shape),
                "loss": last_loss,
            }
        )
    return {
        "config": str(config_path),
        "metadata_path": metadata["metadata_path"],
        "manifest_path": metadata["manifest_path"],
        "split": sampler.split,
        "vocab_size": sampler.vocab_size,
        "batches": batches,
        "small_model_consumed_batch": True,
        "last_loss": last_loss,
    }


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


if __name__ == "__main__":
    raise SystemExit(main())
