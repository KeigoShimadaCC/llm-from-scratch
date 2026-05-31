from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import yaml

from kgpt.checkpoint import load_checkpoint
from kgpt.config import OptimizerConfig, file_sha256
from kgpt.transformer import (
    DecoderOnlyTransformer,
    TransformerDataConfig,
    TransformerModelConfig,
    TransformerSamplingConfig,
    TransformerTokenizerConfig,
)


@dataclass(frozen=True)
class PretrainDataConfig(TransformerDataConfig):
    validation_split: str


@dataclass(frozen=True)
class PretrainTrainingConfig:
    batch_size: int
    train_steps: int
    output_dir: Path
    checkpoint_every: int
    eval_every: int
    sample_every: int
    gradient_accumulation_steps: int
    max_grad_norm: float
    validation_batches: int
    loss_improvement_threshold: float


@dataclass(frozen=True)
class SchedulerConfig:
    warmup_steps: int
    min_lr_factor: float


@dataclass(frozen=True)
class RunBudgetConfig:
    target_steps: int
    target_tokens: int
    target_wall_clock_minutes: int
    hardware: str


@dataclass(frozen=True)
class ScaleGateConfig:
    min_parameters: int
    max_parameters: int
    label: str


@dataclass(frozen=True)
class PretrainConfig:
    run_name: str
    seed: int
    device: str
    dtype: str
    model_name: str
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    model: TransformerModelConfig
    training: PretrainTrainingConfig
    sampling: TransformerSamplingConfig
    data: PretrainDataConfig
    tokenizer: TransformerTokenizerConfig
    run_budget: RunBudgetConfig
    scale: ScaleGateConfig
    sample_prompts: tuple[str, ...]
    source_path: Path
    config_hash: str


@dataclass(frozen=True)
class ResumeState:
    start_step: int
    best_validation_loss: float
    best_step: int
    initial_validation_loss: float


def load_pretrain_config(path: str | Path) -> PretrainConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    model_raw = _required_mapping(raw, "model")
    training_raw = _required_mapping(raw, "training")
    optimizer_raw = _required_mapping(raw, "optimizer")
    scheduler_raw = _required_mapping(raw, "scheduler")
    generation_raw = _required_mapping(raw, "generation")
    data_raw = _required_mapping(raw, "data")
    tokenizer_raw = _required_mapping(raw, "tokenizer")
    budget_raw = _required_mapping(raw, "run_budget")
    scale_raw = raw.get("scale")
    if scale_raw is not None and not isinstance(scale_raw, dict):
        raise ValueError("scale must be a mapping when provided.")
    scale_mapping = scale_raw or {
        "min_parameters": 5_000_000,
        "max_parameters": 20_000_000,
        "label": "kgpt-tiny",
    }
    sample_prompts_raw = raw.get("sample_prompts")
    if not isinstance(sample_prompts_raw, list) or not all(isinstance(item, str) for item in sample_prompts_raw):
        raise ValueError("sample_prompts must be a list of strings.")

    top_k = generation_raw.get("top_k")
    fallback_training_config = tokenizer_raw.get("fallback_training_config")
    config = PretrainConfig(
        run_name=_required_str(raw, "run_name"),
        seed=_required_int(raw, "seed"),
        device=_required_str(raw, "device"),
        dtype=_required_str(raw, "dtype"),
        model_name=_required_str(raw, "model_name"),
        optimizer=OptimizerConfig(
            name=_required_str(optimizer_raw, "name"),
            learning_rate=_positive_float(optimizer_raw, "learning_rate"),
        ),
        scheduler=SchedulerConfig(
            warmup_steps=_non_negative_int(scheduler_raw, "warmup_steps"),
            min_lr_factor=_bounded_float(scheduler_raw, "min_lr_factor", minimum=0.0, maximum=1.0),
        ),
        model=TransformerModelConfig(
            vocab_size=_positive_int(model_raw, "vocab_size"),
            context_length=_positive_int(model_raw, "context_length"),
            embedding_dim=_positive_int(model_raw, "embedding_dim"),
            num_layers=_positive_int(model_raw, "num_layers"),
            num_heads=_positive_int(model_raw, "num_heads"),
            mlp_hidden_dim=_positive_int(model_raw, "mlp_hidden_dim"),
            dropout=_non_negative_float(model_raw, "dropout"),
            tie_embeddings=_required_bool(model_raw, "tie_embeddings"),
        ),
        training=PretrainTrainingConfig(
            batch_size=_positive_int(training_raw, "batch_size"),
            train_steps=_positive_int(training_raw, "train_steps"),
            output_dir=Path(_required_str(training_raw, "output_dir")),
            checkpoint_every=_positive_int(training_raw, "checkpoint_every"),
            eval_every=_positive_int(training_raw, "eval_every"),
            sample_every=_positive_int(training_raw, "sample_every"),
            gradient_accumulation_steps=_positive_int(training_raw, "gradient_accumulation_steps"),
            max_grad_norm=_positive_float(training_raw, "max_grad_norm"),
            validation_batches=_positive_int(training_raw, "validation_batches"),
            loss_improvement_threshold=_bounded_float(
                training_raw, "loss_improvement_threshold", minimum=0.0, maximum=1.0
            ),
        ),
        sampling=TransformerSamplingConfig(
            max_new_tokens=_positive_int(generation_raw, "max_new_tokens"),
            temperature=_non_negative_float(generation_raw, "temperature"),
            top_k=None if top_k in {None, 0} else _positive_int(generation_raw, "top_k"),
        ),
        data=PretrainDataConfig(
            tokenized_config=Path(_required_str(data_raw, "tokenized_config")),
            metadata_path=Path(_required_str(data_raw, "metadata_path")),
            split=_required_str(data_raw, "split"),
            validation_split=_required_str(data_raw, "validation_split"),
        ),
        tokenizer=TransformerTokenizerConfig(
            model_path=Path(_required_str(tokenizer_raw, "model_path")),
            fallback_training_config=Path(fallback_training_config)
            if isinstance(fallback_training_config, str)
            else None,
        ),
        run_budget=RunBudgetConfig(
            target_steps=_positive_int(budget_raw, "target_steps"),
            target_tokens=_positive_int(budget_raw, "target_tokens"),
            target_wall_clock_minutes=_positive_int(budget_raw, "target_wall_clock_minutes"),
            hardware=_required_str(budget_raw, "hardware"),
        ),
        scale=ScaleGateConfig(
            min_parameters=_positive_int(scale_mapping, "min_parameters"),
            max_parameters=_positive_int(scale_mapping, "max_parameters"),
            label=_required_str(scale_mapping, "label"),
        ),
        sample_prompts=tuple(sample_prompts_raw),
        source_path=config_path,
        config_hash=file_sha256(config_path),
    )
    if config.dtype != "float32":
        raise ValueError("PHASE-04A pretraining uses dtype=float32 until mixed precision is justified.")
    if config.optimizer.name != "adamw":
        raise ValueError("PHASE-04A pretraining supports optimizer.name=adamw only.")
    if config.device not in {"cpu", "mps"}:
        raise ValueError("device must be cpu or mps.")
    return config


def pretrain_config_to_dict(config: PretrainConfig) -> dict[str, Any]:
    return {
        "run_name": config.run_name,
        "seed": config.seed,
        "device": config.device,
        "dtype": config.dtype,
        "model_name": config.model_name,
        "optimizer": {
            "name": config.optimizer.name,
            "learning_rate": config.optimizer.learning_rate,
        },
        "scheduler": {
            "warmup_steps": config.scheduler.warmup_steps,
            "min_lr_factor": config.scheduler.min_lr_factor,
        },
        "model": dict(config.model.__dict__),
        "training": {
            "batch_size": config.training.batch_size,
            "train_steps": config.training.train_steps,
            "output_dir": str(config.training.output_dir),
            "checkpoint_every": config.training.checkpoint_every,
            "eval_every": config.training.eval_every,
            "sample_every": config.training.sample_every,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "max_grad_norm": config.training.max_grad_norm,
            "validation_batches": config.training.validation_batches,
            "loss_improvement_threshold": config.training.loss_improvement_threshold,
        },
        "generation": {
            "max_new_tokens": config.sampling.max_new_tokens,
            "temperature": config.sampling.temperature,
            "top_k": config.sampling.top_k,
        },
        "data": {
            "tokenized_config": str(config.data.tokenized_config),
            "metadata_path": str(config.data.metadata_path),
            "split": config.data.split,
            "validation_split": config.data.validation_split,
        },
        "tokenizer": {
            "model_path": str(config.tokenizer.model_path),
            "fallback_training_config": str(config.tokenizer.fallback_training_config)
            if config.tokenizer.fallback_training_config
            else None,
        },
        "run_budget": {
            "target_steps": config.run_budget.target_steps,
            "target_tokens": config.run_budget.target_tokens,
            "target_wall_clock_minutes": config.run_budget.target_wall_clock_minutes,
            "hardware": config.run_budget.hardware,
        },
        "scale": {
            "min_parameters": config.scale.min_parameters,
            "max_parameters": config.scale.max_parameters,
            "label": config.scale.label,
        },
        "sample_prompts": list(config.sample_prompts),
        "source_path": str(config.source_path),
        "config_hash": config.config_hash,
    }


def with_pretrain_overrides(
    config: PretrainConfig,
    *,
    run_name: str | None = None,
    train_steps: int | None = None,
) -> PretrainConfig:
    updated = config
    if run_name is not None:
        updated = replace(updated, run_name=run_name)
    if train_steps is not None:
        updated = replace(
            updated,
            training=replace(updated.training, train_steps=train_steps),
            run_budget=replace(
                updated.run_budget,
                target_steps=train_steps,
                target_tokens=(
                    train_steps
                    * updated.training.batch_size
                    * updated.training.gradient_accumulation_steps
                    * updated.model.context_length
                ),
            ),
        )
    return updated


def learning_rate_for_step(*, base_lr: float, step: int, total_steps: int, scheduler: SchedulerConfig) -> float:
    if step <= 0:
        return 0.0
    if scheduler.warmup_steps > 0 and step <= scheduler.warmup_steps:
        return base_lr * step / scheduler.warmup_steps
    if total_steps <= scheduler.warmup_steps:
        return base_lr
    progress = (step - scheduler.warmup_steps) / (total_steps - scheduler.warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    min_lr = base_lr * scheduler.min_lr_factor
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def create_or_resume_run_dir(config: PretrainConfig, *, resume: bool) -> Path:
    run_dir = config.training.output_dir / config.run_name
    if run_dir.exists() and not resume:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_resume_state(
    *,
    run_dir: Path,
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
) -> ResumeState:
    checkpoint_path = run_dir / "checkpoint_last.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Cannot resume without checkpoint_last.pt: {checkpoint_path}")
    payload = load_checkpoint(checkpoint_path, model=model, optimizer=optimizer, map_location="cpu")
    metadata = payload["metadata"]
    return ResumeState(
        start_step=int(metadata["step"]),
        best_validation_loss=float(metadata["best_validation_loss"]),
        best_step=int(metadata["best_step"]),
        initial_validation_loss=float(metadata["initial_validation_loss"]),
    )


def checkpoint_metadata(
    *,
    config: PretrainConfig,
    run_dir: Path,
    step: int,
    parameter_count: int,
    initial_validation_loss: float,
    current_validation_loss: float,
    best_validation_loss: float,
    best_step: int,
    tokens_seen: int,
    git_commit: str,
    created_at: str,
) -> dict[str, Any]:
    improvement_fraction = loss_improvement_fraction(
        initial_validation_loss=initial_validation_loss,
        validation_loss=current_validation_loss,
    )
    return {
        "config_hash": config.config_hash,
        "config_path": str(run_dir / "config.yaml"),
        "source_config_path": str(config.source_path),
        "model_name": config.model_name,
        "model_config": dict(config.model.__dict__),
        "pretrain_config": pretrain_config_to_dict(config),
        "parameter_count": parameter_count,
        "scale": {
            "label": config.scale.label,
            "min_parameters": config.scale.min_parameters,
            "max_parameters": config.scale.max_parameters,
        },
        "step": step,
        "seed": config.seed,
        "git_commit": git_commit,
        "created_at": created_at,
        "device": config.device,
        "dtype": config.dtype,
        "initial_validation_loss": initial_validation_loss,
        "current_validation_loss": current_validation_loss,
        "best_validation_loss": best_validation_loss,
        "best_step": best_step,
        "loss_improvement_threshold": config.training.loss_improvement_threshold,
        "loss_improvement_fraction": improvement_fraction,
        "loss_improvement_passed": improvement_fraction >= config.training.loss_improvement_threshold,
        "tokens_seen": tokens_seen,
        "resume_supported": True,
    }


def loss_improvement_fraction(*, initial_validation_loss: float, validation_loss: float) -> float:
    if initial_validation_loss <= 0:
        return 0.0
    return (initial_validation_loss - validation_loss) / initial_validation_loss


def perplexity(loss: float) -> float:
    return float(math.exp(min(loss, 20.0)))


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


def _required_bool(raw: dict[str, Any], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Missing boolean config field: {key}")
    return value


def _positive_int(raw: dict[str, Any], key: str) -> int:
    value = _required_int(raw, key)
    if value <= 0:
        raise ValueError(f"Config field must be positive: {key}")
    return value


def _non_negative_int(raw: dict[str, Any], key: str) -> int:
    value = _required_int(raw, key)
    if value < 0:
        raise ValueError(f"Config field must be non-negative: {key}")
    return value


def _positive_float(raw: dict[str, Any], key: str) -> float:
    value = raw.get(key)
    if not isinstance(value, int | float) or float(value) <= 0:
        raise ValueError(f"Config field must be a positive number: {key}")
    return float(value)


def _non_negative_float(raw: dict[str, Any], key: str) -> float:
    value = raw.get(key)
    if not isinstance(value, int | float) or float(value) < 0:
        raise ValueError(f"Config field must be a non-negative number: {key}")
    return float(value)


def _bounded_float(raw: dict[str, Any], key: str, *, minimum: float, maximum: float) -> float:
    value = _non_negative_float(raw, key)
    if value < minimum or value > maximum:
        raise ValueError(f"Config field {key} must be between {minimum} and {maximum}.")
    return value
