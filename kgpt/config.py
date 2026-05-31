from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    learning_rate: float


@dataclass(frozen=True)
class TrainingConfig:
    run_name: str
    seed: int
    device: str
    dtype: str
    model_name: str
    vocab_size: int
    context_length: int
    batch_size: int
    train_steps: int
    optimizer: OptimizerConfig
    output_dir: Path
    checkpoint_every: int
    eval_every: int
    sample_every: int
    tokenizer_id: str
    source_path: Path
    config_hash: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_training_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    optimizer_raw = _required_mapping(raw, "optimizer")
    config = TrainingConfig(
        run_name=_required_str(raw, "run_name"),
        seed=_required_int(raw, "seed"),
        device=_required_str(raw, "device"),
        dtype=_required_str(raw, "dtype"),
        model_name=_required_str(raw, "model_name"),
        vocab_size=_positive_int(raw, "vocab_size"),
        context_length=_positive_int(raw, "context_length"),
        batch_size=_positive_int(raw, "batch_size"),
        train_steps=_positive_int(raw, "train_steps"),
        optimizer=OptimizerConfig(
            name=_required_str(optimizer_raw, "name"),
            learning_rate=_positive_float(optimizer_raw, "learning_rate"),
        ),
        output_dir=Path(_required_str(raw, "output_dir")),
        checkpoint_every=_positive_int(raw, "checkpoint_every"),
        eval_every=_positive_int(raw, "eval_every"),
        sample_every=_positive_int(raw, "sample_every"),
        tokenizer_id=_required_str(raw, "tokenizer_id"),
        source_path=config_path,
        config_hash=file_sha256(config_path),
    )
    if config.dtype != "float32":
        raise ValueError("PHASE-00B dummy training supports dtype=float32 only.")
    if config.device not in {"cpu", "mps"}:
        raise ValueError("device must be cpu or mps")
    return config


def config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    return {
        "run_name": config.run_name,
        "seed": config.seed,
        "device": config.device,
        "dtype": config.dtype,
        "model_name": config.model_name,
        "vocab_size": config.vocab_size,
        "context_length": config.context_length,
        "batch_size": config.batch_size,
        "train_steps": config.train_steps,
        "optimizer": {
            "name": config.optimizer.name,
            "learning_rate": config.optimizer.learning_rate,
        },
        "output_dir": str(config.output_dir),
        "checkpoint_every": config.checkpoint_every,
        "eval_every": config.eval_every,
        "sample_every": config.sample_every,
        "tokenizer_id": config.tokenizer_id,
        "source_path": str(config.source_path),
        "config_hash": config.config_hash,
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


def _positive_int(raw: dict[str, Any], key: str) -> int:
    value = _required_int(raw, key)
    if value <= 0:
        raise ValueError(f"Config field must be positive: {key}")
    return value


def _positive_float(raw: dict[str, Any], key: str) -> float:
    value = raw.get(key)
    if not isinstance(value, int | float) or float(value) <= 0:
        raise ValueError(f"Config field must be a positive number: {key}")
    return float(value)
