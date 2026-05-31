from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from kgpt.char_tokenizer import CharacterTokenizer
from kgpt.checkpoint import load_checkpoint
from kgpt.config import OptimizerConfig, file_sha256


@dataclass(frozen=True)
class MicroCharModelConfig:
    embedding_dim: int
    hidden_dim: int
    dropout: float


@dataclass(frozen=True)
class MicroCharDataConfig:
    name: str
    text: str
    train_fraction: float
    source: str
    license_note: str


@dataclass(frozen=True)
class MicroCharGenerationConfig:
    prompt: str
    max_new_tokens: int
    temperature: float


@dataclass(frozen=True)
class MicroCharConfig:
    run_name: str
    seed: int
    device: str
    dtype: str
    model_name: str
    context_length: int
    batch_size: int
    train_steps: int
    optimizer: OptimizerConfig
    output_dir: Path
    checkpoint_every: int
    eval_every: int
    sample_every: int
    tokenizer_id: str
    model: MicroCharModelConfig
    data: MicroCharDataConfig
    generation: MicroCharGenerationConfig
    overfit_threshold: float
    source_path: Path
    config_hash: str


class CharContextLanguageModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        context_length: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if context_length <= 0:
            raise ValueError("context_length must be positive.")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(context_length * embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.token_embedding(context_tokens)
        return self.network(embedded.flatten(start_dim=1))


def load_micro_char_config(path: str | Path) -> MicroCharConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    optimizer_raw = _required_mapping(raw, "optimizer")
    model_raw = _required_mapping(raw, "model")
    data_raw = _required_mapping(raw, "data")
    generation_raw = _required_mapping(raw, "generation")
    config = MicroCharConfig(
        run_name=_required_str(raw, "run_name"),
        seed=_required_int(raw, "seed"),
        device=_required_str(raw, "device"),
        dtype=_required_str(raw, "dtype"),
        model_name=_required_str(raw, "model_name"),
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
        model=MicroCharModelConfig(
            embedding_dim=_positive_int(model_raw, "embedding_dim"),
            hidden_dim=_positive_int(model_raw, "hidden_dim"),
            dropout=_non_negative_float(model_raw, "dropout"),
        ),
        data=MicroCharDataConfig(
            name=_required_str(data_raw, "name"),
            text=_required_str(data_raw, "text"),
            train_fraction=_fraction(data_raw, "train_fraction"),
            source=_required_str(data_raw, "source"),
            license_note=_required_str(data_raw, "license_note"),
        ),
        generation=MicroCharGenerationConfig(
            prompt=_required_str(generation_raw, "prompt"),
            max_new_tokens=_positive_int(generation_raw, "max_new_tokens"),
            temperature=_non_negative_float(generation_raw, "temperature"),
        ),
        overfit_threshold=_positive_float(raw, "overfit_threshold"),
        source_path=config_path,
        config_hash=file_sha256(config_path),
    )
    if config.dtype != "float32":
        raise ValueError("PHASE-01A micro character training supports dtype=float32 only.")
    if config.device not in {"cpu", "mps"}:
        raise ValueError("device must be cpu or mps")
    if config.optimizer.name != "adamw":
        raise ValueError("PHASE-01A micro character training supports optimizer.name=adamw only.")
    return config


def config_to_dict(config: MicroCharConfig) -> dict[str, Any]:
    return {
        "run_name": config.run_name,
        "seed": config.seed,
        "device": config.device,
        "dtype": config.dtype,
        "model_name": config.model_name,
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
        "model": {
            "embedding_dim": config.model.embedding_dim,
            "hidden_dim": config.model.hidden_dim,
            "dropout": config.model.dropout,
        },
        "data": {
            "name": config.data.name,
            "train_fraction": config.data.train_fraction,
            "source": config.data.source,
            "license_note": config.data.license_note,
            "characters": len(config.data.text),
        },
        "generation": {
            "prompt": config.generation.prompt,
            "max_new_tokens": config.generation.max_new_tokens,
            "temperature": config.generation.temperature,
        },
        "overfit_threshold": config.overfit_threshold,
        "source_path": str(config.source_path),
        "config_hash": config.config_hash,
    }


def with_overrides(
    config: MicroCharConfig,
    *,
    run_name: str | None = None,
    train_steps: int | None = None,
) -> MicroCharConfig:
    return MicroCharConfig(
        run_name=run_name or config.run_name,
        seed=config.seed,
        device=config.device,
        dtype=config.dtype,
        model_name=config.model_name,
        context_length=config.context_length,
        batch_size=config.batch_size,
        train_steps=train_steps or config.train_steps,
        optimizer=config.optimizer,
        output_dir=config.output_dir,
        checkpoint_every=config.checkpoint_every,
        eval_every=config.eval_every,
        sample_every=config.sample_every,
        tokenizer_id=config.tokenizer_id,
        model=config.model,
        data=config.data,
        generation=config.generation,
        overfit_threshold=config.overfit_threshold,
        source_path=config.source_path,
        config_hash=config.config_hash,
    )


def build_next_char_dataset(
    token_ids: list[int],
    *,
    context_length: int,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(token_ids) < 2:
        raise ValueError("At least two token ids are required for next-character training.")

    contexts: list[list[int]] = []
    targets: list[int] = []
    for target_position in range(1, len(token_ids)):
        start_position = max(0, target_position - context_length)
        context = token_ids[start_position:target_position]
        padded_context = [pad_token_id] * (context_length - len(context)) + context
        contexts.append(padded_context)
        targets.append(token_ids[target_position])
    return torch.tensor(contexts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def split_dataset(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    train_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must contain the same number of examples.")
    if inputs.shape[0] < 2:
        raise ValueError("At least two examples are required for a train/validation split.")
    split_index = int(inputs.shape[0] * train_fraction)
    split_index = min(max(split_index, 1), inputs.shape[0] - 1)
    return inputs[:split_index], targets[:split_index], inputs[split_index:], targets[split_index:]


def evaluate_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        logits = model(inputs.to(device))
        loss = criterion(logits, targets.to(device))
    return float(loss.detach().cpu())


def generate_text(
    *,
    model: CharContextLanguageModel,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_new_tokens: int,
    seed: int,
    temperature: float,
    greedy: bool,
    device: torch.device,
) -> str:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise ValueError("Prompt must encode to at least one token.")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    model.eval()
    for _ in range(max_new_tokens):
        context = token_ids[-model.context_length :]
        context = [0] * (model.context_length - len(context)) + context
        context_tensor = torch.tensor([context], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(context_tensor)[0].detach().cpu()

        if greedy or temperature == 0:
            next_token_id = int(torch.argmax(logits).item())
        else:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            next_token_id = int(torch.multinomial(probabilities, num_samples=1, generator=generator).item())
        token_ids.append(next_token_id)
    return tokenizer.decode(token_ids)


def load_micro_char_checkpoint(
    path: str | Path,
    *,
    map_location: str = "cpu",
) -> tuple[CharContextLanguageModel, CharacterTokenizer, dict[str, Any]]:
    payload = load_checkpoint(path, map_location=map_location)
    metadata = payload["metadata"]
    model_config = metadata.get("model_config")
    tokenizer_payload = metadata.get("tokenizer")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint metadata is missing model_config.")
    if not isinstance(tokenizer_payload, dict):
        raise ValueError("Checkpoint metadata is missing tokenizer.")

    tokenizer = CharacterTokenizer.from_dict(tokenizer_payload)
    model = CharContextLanguageModel(
        vocab_size=tokenizer.vocab_size,
        context_length=int(model_config["context_length"]),
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        dropout=float(model_config.get("dropout", 0.0)),
    )
    model.load_state_dict(payload["model_state"])
    return model, tokenizer, metadata


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


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


def _fraction(raw: dict[str, Any], key: str) -> float:
    value = _positive_float(raw, key)
    if value >= 1:
        raise ValueError(f"Config field must be less than 1: {key}")
    return value
