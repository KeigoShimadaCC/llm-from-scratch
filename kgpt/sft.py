from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import yaml

from kgpt.byte_bpe import ByteBPETokenizer
from kgpt.config import file_sha256
from kgpt.pretrain import load_pretrain_config
from kgpt.token_data import normalize_text
from kgpt.transformer import TransformerSamplingConfig

IGNORE_INDEX = -100


@dataclass(frozen=True)
class InstructionRecord:
    record_id: str
    instruction: str
    response: str


@dataclass(frozen=True)
class PromptTemplateConfig:
    version: str
    pattern: str
    prompt_pattern: str


@dataclass(frozen=True)
class SFTDatasetConfig:
    source_name: str
    license: str
    split_seed: int
    validation_fraction: float
    dedup_strategy: str
    contamination_notes: str
    response_only_loss: bool
    manifest_path: Path
    records: tuple[InstructionRecord, ...]


@dataclass(frozen=True)
class SFTTrainingConfig:
    batch_size: int
    train_steps: int
    output_dir: Path
    checkpoint_every: int
    eval_every: int
    sample_every: int
    learning_rate: float
    max_grad_norm: float


@dataclass(frozen=True)
class SFTBaseConfig:
    config: Path
    checkpoint: Path
    bootstrap_if_missing: bool
    bootstrap_max_steps: int
    bootstrap_run_name: str


@dataclass(frozen=True)
class SFTConfig:
    run_name: str
    seed: int
    device: str
    dtype: str
    base: SFTBaseConfig
    prompt_template: PromptTemplateConfig
    dataset: SFTDatasetConfig
    training: SFTTrainingConfig
    sampling: TransformerSamplingConfig
    source_path: Path
    config_hash: str


@dataclass(frozen=True)
class SFTExample:
    input_ids: torch.Tensor
    targets: torch.Tensor
    record_id: str


def load_sft_config(path: str | Path) -> SFTConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"SFT config must be a mapping: {config_path}")

    base_raw = _required_mapping(raw, "base")
    template_raw = _required_mapping(raw, "prompt_template")
    dataset_raw = _required_mapping(raw, "dataset")
    training_raw = _required_mapping(raw, "training")
    generation_raw = _required_mapping(raw, "generation")
    top_k = generation_raw.get("top_k")
    records_raw = dataset_raw.get("records")
    if not isinstance(records_raw, list) or not records_raw:
        raise ValueError("dataset.records must be a non-empty list.")
    records = tuple(
        InstructionRecord(
            record_id=_required_str(item, "id"),
            instruction=_required_str(item, "instruction"),
            response=_required_str(item, "response"),
        )
        for item in records_raw
        if isinstance(item, dict)
    )
    if len(records) != len(records_raw):
        raise ValueError("dataset.records entries must be mappings.")

    config = SFTConfig(
        run_name=_required_str(raw, "run_name"),
        seed=_required_int(raw, "seed"),
        device=_required_str(raw, "device"),
        dtype=_required_str(raw, "dtype"),
        base=SFTBaseConfig(
            config=Path(_required_str(base_raw, "config")),
            checkpoint=Path(_required_str(base_raw, "checkpoint")),
            bootstrap_if_missing=_required_bool(base_raw, "bootstrap_if_missing"),
            bootstrap_max_steps=_positive_int(base_raw, "bootstrap_max_steps"),
            bootstrap_run_name=_required_str(base_raw, "bootstrap_run_name"),
        ),
        prompt_template=PromptTemplateConfig(
            version=_required_str(template_raw, "version"),
            pattern=_required_str(template_raw, "pattern"),
            prompt_pattern=_required_str(template_raw, "prompt_pattern"),
        ),
        dataset=SFTDatasetConfig(
            source_name=_required_str(dataset_raw, "source_name"),
            license=_required_str(dataset_raw, "license"),
            split_seed=_required_int(dataset_raw, "split_seed"),
            validation_fraction=_positive_float(dataset_raw, "validation_fraction"),
            dedup_strategy=_required_str(dataset_raw, "dedup_strategy"),
            contamination_notes=_required_str(dataset_raw, "contamination_notes"),
            response_only_loss=_required_bool(dataset_raw, "response_only_loss"),
            manifest_path=Path(_required_str(dataset_raw, "manifest_path")),
            records=records,
        ),
        training=SFTTrainingConfig(
            batch_size=_positive_int(training_raw, "batch_size"),
            train_steps=_positive_int(training_raw, "train_steps"),
            output_dir=Path(_required_str(training_raw, "output_dir")),
            checkpoint_every=_positive_int(training_raw, "checkpoint_every"),
            eval_every=_positive_int(training_raw, "eval_every"),
            sample_every=_positive_int(training_raw, "sample_every"),
            learning_rate=_positive_float(training_raw, "learning_rate"),
            max_grad_norm=_positive_float(training_raw, "max_grad_norm"),
        ),
        sampling=TransformerSamplingConfig(
            max_new_tokens=_positive_int(generation_raw, "max_new_tokens"),
            temperature=_non_negative_float(generation_raw, "temperature"),
            top_k=None if top_k in {None, 0} else _positive_int(generation_raw, "top_k"),
        ),
        source_path=config_path,
        config_hash=file_sha256(config_path),
    )
    if config.dtype != "float32":
        raise ValueError("PHASE-06A SFT uses dtype=float32.")
    if config.device not in {"cpu", "mps"}:
        raise ValueError("device must be cpu or mps.")
    _deduplicate_records(config.dataset.records)
    return config


def with_sft_overrides(
    config: SFTConfig,
    *,
    run_name: str | None = None,
    train_steps: int | None = None,
) -> SFTConfig:
    updated = config
    if run_name is not None:
        updated = replace(updated, run_name=run_name)
    if train_steps is not None:
        updated = replace(updated, training=replace(updated.training, train_steps=train_steps))
    return updated


def format_prompt(template: PromptTemplateConfig, instruction: str) -> str:
    return template.prompt_pattern.format(instruction=instruction)


def format_example(template: PromptTemplateConfig, record: InstructionRecord) -> tuple[str, str]:
    prompt = format_prompt(template, record.instruction)
    full_text = template.pattern.format(instruction=record.instruction, response=record.response)
    return prompt, full_text


def split_instruction_records(
    records: tuple[InstructionRecord, ...],
    *,
    validation_fraction: float,
    seed: int,
) -> dict[str, tuple[InstructionRecord, ...]]:
    ordered = sorted(
        records,
        key=lambda record: hashlib.sha256(f"{seed}:{record.record_id}:{record.instruction}".encode()).hexdigest(),
    )
    validation_count = int(round(len(ordered) * validation_fraction))
    validation_count = min(max(validation_count, 1), len(ordered) - 1)
    return {
        "validation": tuple(ordered[:validation_count]),
        "train": tuple(ordered[validation_count:]),
    }


def build_sft_examples(
    *,
    records: tuple[InstructionRecord, ...],
    tokenizer: ByteBPETokenizer,
    template: PromptTemplateConfig,
    context_length: int,
    response_only_loss: bool,
) -> list[SFTExample]:
    examples: list[SFTExample] = []
    for record in records:
        prompt, full_text = format_example(template, record)
        prompt_ids = tokenizer.encode(prompt, add_bos=True)
        token_ids = tokenizer.encode(full_text, add_bos=True, add_eos=True)
        if len(token_ids) < 2:
            raise ValueError(f"SFT record is too short: {record.record_id}")
        if len(token_ids) > context_length + 1:
            raise ValueError(f"SFT record exceeds context length {context_length}: {record.record_id}")
        inputs = token_ids[:-1]
        targets = token_ids[1:]
        if response_only_loss:
            prompt_target_tokens = max(len(prompt_ids) - 1, 0)
            targets[:prompt_target_tokens] = [IGNORE_INDEX] * prompt_target_tokens
        pad_count = context_length - len(inputs)
        input_ids = inputs + [tokenizer.pad_token_id] * pad_count
        target_ids = targets + [IGNORE_INDEX] * pad_count
        examples.append(
            SFTExample(
                input_ids=torch.tensor(input_ids, dtype=torch.long),
                targets=torch.tensor(target_ids, dtype=torch.long),
                record_id=record.record_id,
            )
        )
    return examples


def sft_config_to_dict(config: SFTConfig) -> dict[str, Any]:
    return {
        "run_name": config.run_name,
        "seed": config.seed,
        "device": config.device,
        "dtype": config.dtype,
        "base": {
            "config": str(config.base.config),
            "checkpoint": str(config.base.checkpoint),
            "bootstrap_if_missing": config.base.bootstrap_if_missing,
            "bootstrap_max_steps": config.base.bootstrap_max_steps,
            "bootstrap_run_name": config.base.bootstrap_run_name,
        },
        "prompt_template": {
            "version": config.prompt_template.version,
            "pattern": config.prompt_template.pattern,
            "prompt_pattern": config.prompt_template.prompt_pattern,
        },
        "dataset": {
            "source_name": config.dataset.source_name,
            "license": config.dataset.license,
            "split_seed": config.dataset.split_seed,
            "validation_fraction": config.dataset.validation_fraction,
            "dedup_strategy": config.dataset.dedup_strategy,
            "contamination_notes": config.dataset.contamination_notes,
            "response_only_loss": config.dataset.response_only_loss,
            "manifest_path": str(config.dataset.manifest_path),
            "record_count": len(config.dataset.records),
        },
        "training": {
            "batch_size": config.training.batch_size,
            "train_steps": config.training.train_steps,
            "output_dir": str(config.training.output_dir),
            "checkpoint_every": config.training.checkpoint_every,
            "eval_every": config.training.eval_every,
            "sample_every": config.training.sample_every,
            "learning_rate": config.training.learning_rate,
            "max_grad_norm": config.training.max_grad_norm,
        },
        "generation": {
            "max_new_tokens": config.sampling.max_new_tokens,
            "temperature": config.sampling.temperature,
            "top_k": config.sampling.top_k,
        },
        "source_path": str(config.source_path),
        "config_hash": config.config_hash,
    }


def write_instruction_manifest(config: SFTConfig, splits: dict[str, tuple[InstructionRecord, ...]]) -> None:
    manifest = {
        "schema_version": 1,
        "source_name": config.dataset.source_name,
        "license": config.dataset.license,
        "prompt_template_version": config.prompt_template.version,
        "dedup_strategy": config.dataset.dedup_strategy,
        "contamination_notes": config.dataset.contamination_notes,
        "response_only_loss": config.dataset.response_only_loss,
        "split_seed": config.dataset.split_seed,
        "validation_fraction": config.dataset.validation_fraction,
        "splits": {
            name: {
                "record_count": len(records),
                "record_ids": [record.record_id for record in records],
            }
            for name, records in splits.items()
        },
        "record_count": len(config.dataset.records),
    }
    config.dataset.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.dataset.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf8")


def load_base_pretrain_config(config: SFTConfig):
    return load_pretrain_config(config.base.config)


def _deduplicate_records(records: tuple[InstructionRecord, ...]) -> None:
    seen: set[str] = set()
    for record in records:
        digest = hashlib.sha256(
            f"{normalize_text(record.instruction)}\n{normalize_text(record.response)}".encode()
        ).hexdigest()
        if digest in seen:
            raise ValueError(f"Duplicate SFT instruction/response pair: {record.record_id}")
        seen.add(digest)


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
