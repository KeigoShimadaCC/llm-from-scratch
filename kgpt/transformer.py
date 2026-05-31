from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from kgpt.byte_bpe import ByteBPETokenizer
from kgpt.checkpoint import load_checkpoint
from kgpt.config import OptimizerConfig, file_sha256


@dataclass(frozen=True)
class TransformerModelConfig:
    vocab_size: int
    context_length: int
    embedding_dim: int
    num_layers: int
    num_heads: int
    mlp_hidden_dim: int
    dropout: float
    tie_embeddings: bool


@dataclass(frozen=True)
class TransformerTrainingConfig:
    batch_size: int
    train_steps: int
    output_dir: Path
    checkpoint_every: int
    eval_every: int


@dataclass(frozen=True)
class TransformerSamplingConfig:
    max_new_tokens: int
    temperature: float
    top_k: int | None


@dataclass(frozen=True)
class TransformerDataConfig:
    tokenized_config: Path
    metadata_path: Path
    split: str


@dataclass(frozen=True)
class TransformerTokenizerConfig:
    model_path: Path
    fallback_training_config: Path | None


@dataclass(frozen=True)
class TransformerExperimentConfig:
    run_name: str
    seed: int
    device: str
    dtype: str
    model_name: str
    optimizer: OptimizerConfig
    model: TransformerModelConfig
    training: TransformerTrainingConfig
    sampling: TransformerSamplingConfig
    data: TransformerDataConfig
    tokenizer: TransformerTokenizerConfig
    source_path: Path
    config_hash: str


@dataclass(frozen=True)
class TransformerOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None


def causal_mask(sequence_length: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    return torch.tril(torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=device)).view(
        1, 1, sequence_length, sequence_length
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerModelConfig) -> None:
        super().__init__()
        if config.embedding_dim % config.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.query_key_value = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = hidden_states.shape
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~causal_mask(sequence_length, device=hidden_states.device), float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim)
        return self.projection(attended)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, config.mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(config.mlp_hidden_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(self.attention_norm(hidden_states))
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: TransformerModelConfig) -> None:
        super().__init__()
        _validate_model_config(config)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None) -> TransformerOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time].")
        if input_ids.shape[1] > self.config.context_length:
            raise ValueError("input sequence length exceeds configured context_length.")

        batch_size, sequence_length = input_ids.shape
        positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden_states = self.drop(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        logits = self.lm_head(self.final_norm(hidden_states))
        loss = shifted_cross_entropy(logits, targets) if targets is not None else None
        return TransformerOutput(logits=logits, loss=loss)


def shifted_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, time, vocab].")
    if targets.shape != logits.shape[:2]:
        raise ValueError("targets must have shape [batch, time] matching logits.")
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def shift_tokens_for_next_token_loss(token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [batch, time].")
    if token_ids.shape[1] < 2:
        raise ValueError("At least two time steps are required for next-token loss.")
    return token_ids[:, :-1], token_ids[:, 1:]


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def resolve_device(requested: str, *, mps_available: bool | None = None) -> torch.device:
    if requested not in {"cpu", "mps"}:
        raise ValueError("device must be cpu or mps.")
    if requested == "mps":
        available = torch.backends.mps.is_available() if mps_available is None else mps_available
        if available:
            return torch.device("mps")
    return torch.device("cpu")


def generate_tokens(
    *,
    model: DecoderOnlyTransformer,
    input_ids: list[int] | torch.Tensor,
    max_new_tokens: int,
    seed: int,
    temperature: float,
    top_k: int | None = None,
    eos_token_id: int | None = None,
    device: torch.device | None = None,
) -> list[int]:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")
    if isinstance(input_ids, torch.Tensor):
        tokens = [int(item) for item in input_ids.flatten().tolist()]
    else:
        tokens = list(input_ids)
    if not tokens:
        raise ValueError("input_ids must contain at least one token.")

    run_device = device or next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model.eval()
    for _ in range(max_new_tokens):
        context = tokens[-model.config.context_length :]
        context_tensor = torch.tensor([context], dtype=torch.long, device=run_device)
        with torch.no_grad():
            logits = model(context_tensor).logits[0, -1].detach().cpu()
        next_token_id = _sample_next_token(logits, temperature=temperature, top_k=top_k, generator=generator)
        tokens.append(next_token_id)
        if eos_token_id is not None and next_token_id == eos_token_id:
            break
    return tokens


def load_transformer_experiment_config(path: str | Path) -> TransformerExperimentConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    model_raw = _required_mapping(raw, "model")
    training_raw = _required_mapping(raw, "training")
    optimizer_raw = _required_mapping(raw, "optimizer")
    sampling_raw = _required_mapping(raw, "generation")
    data_raw = _required_mapping(raw, "data")
    tokenizer_raw = _required_mapping(raw, "tokenizer")
    top_k = sampling_raw.get("top_k")
    fallback_training_config = tokenizer_raw.get("fallback_training_config")
    config = TransformerExperimentConfig(
        run_name=_required_str(raw, "run_name"),
        seed=_required_int(raw, "seed"),
        device=_required_str(raw, "device"),
        dtype=_required_str(raw, "dtype"),
        model_name=_required_str(raw, "model_name"),
        optimizer=OptimizerConfig(
            name=_required_str(optimizer_raw, "name"),
            learning_rate=_positive_float(optimizer_raw, "learning_rate"),
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
        training=TransformerTrainingConfig(
            batch_size=_positive_int(training_raw, "batch_size"),
            train_steps=_positive_int(training_raw, "train_steps"),
            output_dir=Path(_required_str(training_raw, "output_dir")),
            checkpoint_every=_positive_int(training_raw, "checkpoint_every"),
            eval_every=_positive_int(training_raw, "eval_every"),
        ),
        sampling=TransformerSamplingConfig(
            max_new_tokens=_positive_int(sampling_raw, "max_new_tokens"),
            temperature=_non_negative_float(sampling_raw, "temperature"),
            top_k=None if top_k in {None, 0} else _positive_int(sampling_raw, "top_k"),
        ),
        data=TransformerDataConfig(
            tokenized_config=Path(_required_str(data_raw, "tokenized_config")),
            metadata_path=Path(_required_str(data_raw, "metadata_path")),
            split=_required_str(data_raw, "split"),
        ),
        tokenizer=TransformerTokenizerConfig(
            model_path=Path(_required_str(tokenizer_raw, "model_path")),
            fallback_training_config=Path(fallback_training_config)
            if isinstance(fallback_training_config, str)
            else None,
        ),
        source_path=config_path,
        config_hash=file_sha256(config_path),
    )
    if config.dtype != "float32":
        raise ValueError("PHASE-03A Transformer uses dtype=float32 until mixed precision is justified.")
    if config.optimizer.name != "adamw":
        raise ValueError("PHASE-03A Transformer smoke supports optimizer.name=adamw only.")
    _validate_model_config(config.model)
    return config


def transformer_config_to_dict(config: TransformerExperimentConfig) -> dict[str, Any]:
    return {
        "run_name": config.run_name,
        "seed": config.seed,
        "device": config.device,
        "dtype": config.dtype,
        "model_name": config.model_name,
        "optimizer": {"name": config.optimizer.name, "learning_rate": config.optimizer.learning_rate},
        "model": config.model.__dict__,
        "training": {
            "batch_size": config.training.batch_size,
            "train_steps": config.training.train_steps,
            "output_dir": str(config.training.output_dir),
            "checkpoint_every": config.training.checkpoint_every,
            "eval_every": config.training.eval_every,
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
        },
        "tokenizer": {
            "model_path": str(config.tokenizer.model_path),
            "fallback_training_config": str(config.tokenizer.fallback_training_config)
            if config.tokenizer.fallback_training_config
            else None,
        },
        "source_path": str(config.source_path),
        "config_hash": config.config_hash,
    }


def load_tokenizer_for_config(config: TransformerExperimentConfig) -> ByteBPETokenizer:
    if not config.tokenizer.model_path.is_file():
        if not config.tokenizer.fallback_training_config:
            raise FileNotFoundError(f"Tokenizer model not found: {config.tokenizer.model_path}")
        from tokenizer.train_report import generate_tokenizer_report

        generate_tokenizer_report(
            config_path=config.tokenizer.fallback_training_config,
            output_path=Path("docs/tokenizer_report.md"),
        )
    return ByteBPETokenizer.load(config.tokenizer.model_path)


def load_transformer_checkpoint(
    path: str | Path,
    *,
    config: TransformerExperimentConfig,
    map_location: str = "cpu",
) -> tuple[DecoderOnlyTransformer, dict[str, Any]]:
    model = DecoderOnlyTransformer(config.model)
    payload = load_checkpoint(path, model=model, map_location=map_location)
    return model, payload["metadata"]


def _sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    generator: torch.Generator,
) -> int:
    if temperature == 0:
        return int(torch.argmax(logits).item())
    scaled = logits / temperature
    if top_k is not None:
        values, _indices = torch.topk(scaled, k=min(top_k, scaled.shape[-1]))
        floor = values[-1]
        scaled = torch.where(scaled < floor, torch.full_like(scaled, float("-inf")), scaled)
    probabilities = F.softmax(scaled, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1, generator=generator).item())


def _validate_model_config(config: TransformerModelConfig) -> None:
    if config.vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")
    if config.context_length <= 0:
        raise ValueError("context_length must be positive.")
    if config.embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive.")
    if config.num_layers <= 0:
        raise ValueError("num_layers must be positive.")
    if config.num_heads <= 0:
        raise ValueError("num_heads must be positive.")
    if config.embedding_dim % config.num_heads != 0:
        raise ValueError("embedding_dim must be divisible by num_heads.")
    if config.mlp_hidden_dim <= 0:
        raise ValueError("mlp_hidden_dim must be positive.")
    if config.dropout < 0:
        raise ValueError("dropout must be non-negative.")


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
