from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from kgpt.byte_bpe import ByteBPETokenizer
from kgpt.checkpoint import load_checkpoint
from kgpt.pretrain import PretrainConfig, load_pretrain_config
from kgpt.seed import seed_everything
from kgpt.transformer import (
    DecoderOnlyTransformer,
    TransformerExperimentConfig,
    generate_tokens,
    load_tokenizer_for_config,
    load_transformer_experiment_config,
    resolve_device,
)
from train.transformer_smoke import create_run_dir, run_transformer_smoke_training


@dataclass(frozen=True)
class InferenceConfig:
    source_path: Path
    run_name: str
    model_kind: str
    model_config: Path
    checkpoint: Path
    bootstrap_if_missing: bool
    bootstrap_max_steps: int
    device: str
    dtype: str
    generation: dict[str, Any]
    chat_template: str
    benchmark: dict[str, Any]
    mlx: dict[str, Any]


@dataclass(frozen=True)
class LoadedInferenceModel:
    model: DecoderOnlyTransformer
    tokenizer: Any
    checkpoint_path: Path
    metadata: dict[str, Any]
    experiment_config: TransformerExperimentConfig | PretrainConfig
    device: torch.device


def load_inference_config(path: str | Path) -> InferenceConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Inference config must be a mapping: {config_path}")
    if _is_legacy_transformer_config(raw):
        transformer_config = load_transformer_experiment_config(config_path)
        return InferenceConfig(
            source_path=config_path,
            run_name=transformer_config.run_name,
            model_kind="transformer_smoke",
            model_config=config_path,
            checkpoint=transformer_config.training.output_dir
            / transformer_config.run_name
            / "checkpoint_last.pt",
            bootstrap_if_missing=False,
            bootstrap_max_steps=transformer_config.training.train_steps,
            device=transformer_config.device,
            dtype=transformer_config.dtype,
            generation={
                "max_new_tokens": transformer_config.sampling.max_new_tokens,
                "temperature": transformer_config.sampling.temperature,
                "top_k": transformer_config.sampling.top_k,
                "top_p": None,
                "repetition_penalty": 1.0,
                "use_cache": False,
                "stop_strings": [],
                "stop_token_ids": [],
            },
            chat_template="Q:{instruction}\nA:",
            benchmark={"devices": ["cpu"], "warmup_runs": 1, "measured_runs": 1, "prompts": ["hello"]},
            mlx={"status": "not_configured"},
        )

    model_raw = _required_mapping(raw, "model")
    generation_raw = raw.get("generation") if isinstance(raw.get("generation"), dict) else {}
    benchmark_raw = raw.get("benchmark") if isinstance(raw.get("benchmark"), dict) else {}
    chat_raw = raw.get("chat") if isinstance(raw.get("chat"), dict) else {}
    bootstrap_raw = raw.get("bootstrap") if isinstance(raw.get("bootstrap"), dict) else {}
    mlx_raw = raw.get("mlx") if isinstance(raw.get("mlx"), dict) else {}
    return InferenceConfig(
        source_path=config_path,
        run_name=str(raw.get("run_name", config_path.stem)),
        model_kind=_required_str(model_raw, "kind"),
        model_config=Path(_required_str(model_raw, "config")),
        checkpoint=Path(_required_str(model_raw, "checkpoint")),
        bootstrap_if_missing=bool(bootstrap_raw.get("if_missing", False)),
        bootstrap_max_steps=int(bootstrap_raw.get("max_steps", 20)),
        device=str(raw.get("device", "cpu")),
        dtype=str(raw.get("dtype", "float32")),
        generation=_generation_defaults(generation_raw),
        chat_template=str(chat_raw.get("template", "Q:{instruction}\nA:")),
        benchmark={
            "devices": list(benchmark_raw.get("devices", ["cpu", "mps", "mlx"])),
            "warmup_runs": int(benchmark_raw.get("warmup_runs", 1)),
            "measured_runs": int(benchmark_raw.get("measured_runs", 2)),
            "prompts": list(benchmark_raw.get("prompts", ["hello"])),
        },
        mlx={
            "required": bool(mlx_raw.get("required", False)),
            "deferral_doc": str(mlx_raw.get("deferral_doc", "docs/phase08a_mlx_deferral.md")),
        },
    )


def load_model_for_inference(
    *,
    config: InferenceConfig,
    checkpoint_override: Path | None = None,
    device_override: str | None = None,
    tokenizer_override: Path | None = None,
) -> LoadedInferenceModel:
    if config.model_kind == "pretrain":
        experiment_config = load_pretrain_config(config.model_config)
        checkpoint_path = checkpoint_override or config.checkpoint
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        tokenizer = (
            ByteBPETokenizer.load(tokenizer_override)
            if tokenizer_override
            else load_tokenizer_for_config(experiment_config)
        )
        model = DecoderOnlyTransformer(experiment_config.model)
        payload = load_checkpoint(checkpoint_path, model=model, map_location="cpu")
        device = resolve_device(device_override or config.device)
        model.to(device)
        model.eval()
        return LoadedInferenceModel(
            model=model,
            tokenizer=tokenizer,
            checkpoint_path=checkpoint_path,
            metadata=payload["metadata"],
            experiment_config=experiment_config,
            device=device,
        )

    if config.model_kind != "transformer_smoke":
        raise ValueError(f"Unsupported inference model kind: {config.model_kind}")
    experiment_config = load_transformer_experiment_config(config.model_config)
    checkpoint_path = checkpoint_override or config.checkpoint
    ensure_checkpoint(config=config, experiment_config=experiment_config, checkpoint_path=checkpoint_path)
    tokenizer = (
        ByteBPETokenizer.load(tokenizer_override)
        if tokenizer_override
        else load_tokenizer_for_config(experiment_config)
    )
    model = DecoderOnlyTransformer(experiment_config.model)
    payload = load_checkpoint(checkpoint_path, model=model, map_location="cpu")
    device = resolve_device(device_override or config.device)
    model.to(device)
    model.eval()
    return LoadedInferenceModel(
        model=model,
        tokenizer=tokenizer,
        checkpoint_path=checkpoint_path,
        metadata=payload["metadata"],
        experiment_config=experiment_config,
        device=device,
    )


def ensure_checkpoint(
    *,
    config: InferenceConfig,
    experiment_config: TransformerExperimentConfig,
    checkpoint_path: Path,
) -> None:
    if checkpoint_path.is_file():
        return
    if not config.bootstrap_if_missing:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if checkpoint_path != experiment_config.training.output_dir / experiment_config.run_name / "checkpoint_last.pt":
        raise FileNotFoundError(
            "Bootstrap can only create the checkpoint path implied by the model config: "
            f"{checkpoint_path}"
        )
    run_dir = create_run_dir(experiment_config.run_name, experiment_config.training.output_dir)
    run_transformer_smoke_training(
        config=experiment_config,
        run_dir=run_dir,
        max_steps=config.bootstrap_max_steps,
    )


def generate_completion(
    *,
    loaded: LoadedInferenceModel,
    prompt: str,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float,
    stop_strings: list[str],
    stop_token_ids: set[int],
    use_cache: bool,
) -> dict[str, Any]:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    seed_everything(seed)
    prompt_token_ids = loaded.tokenizer.encode(prompt, add_bos=True)
    started = time.perf_counter()
    generated_token_ids = generate_tokens(
        model=loaded.model,
        input_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        seed=seed,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=loaded.tokenizer.eos_token_id,
        stop_token_ids=stop_token_ids,
        device=loaded.device,
        use_cache=use_cache,
    )
    elapsed = max(time.perf_counter() - started, 1e-9)
    generated_text = loaded.tokenizer.decode(generated_token_ids)
    generated_text, stop_reason = truncate_at_stop_string(
        generated_text=generated_text,
        prompt=prompt,
        stop_strings=stop_strings,
    )
    new_text = completion_after_prompt(generated_text=generated_text, prompt=prompt)
    new_tokens = max(len(generated_token_ids) - len(prompt_token_ids), 0)
    return {
        "checkpoint": str(loaded.checkpoint_path),
        "prompt": prompt,
        "generated_text": generated_text,
        "new_text": new_text,
        "seed": seed,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "prompt_token_count": len(prompt_token_ids),
        "generated_token_count": len(generated_token_ids),
        "new_token_count": new_tokens,
        "tokens_per_sec": new_tokens / elapsed,
        "latency_sec": elapsed,
        "device": str(loaded.device),
        "use_cache": use_cache,
        "stop_reason": stop_reason,
        "model_name": loaded.metadata.get("model_name"),
        "step": loaded.metadata.get("step"),
    }


def config_generation_value(
    *,
    config: InferenceConfig,
    name: str,
    override: Any,
) -> Any:
    return override if override is not None else config.generation[name]


def normalize_top_k(value: int | None) -> int | None:
    if value == 0:
        return None
    if value is not None and value < 0:
        raise ValueError("top_k must be non-negative.")
    return value


def normalize_stop_token_ids(values: list[int] | None, config: InferenceConfig) -> set[int]:
    return {int(item) for item in config.generation["stop_token_ids"]} | set(values or [])


def truncate_at_stop_string(
    *,
    generated_text: str,
    prompt: str,
    stop_strings: list[str],
) -> tuple[str, str | None]:
    if not stop_strings:
        return generated_text, None
    new_text = completion_after_prompt(generated_text=generated_text, prompt=prompt)
    earliest: tuple[int, str] | None = None
    for stop_string in stop_strings:
        if not stop_string:
            continue
        index = new_text.find(stop_string)
        if index >= 0 and (earliest is None or index < earliest[0]):
            earliest = (index, stop_string)
    if earliest is None:
        return generated_text, None
    prefix = generated_text[: len(generated_text) - len(new_text)]
    return prefix + new_text[: earliest[0]], f"stop_string:{earliest[1]}"


def completion_after_prompt(*, generated_text: str, prompt: str) -> str:
    if generated_text.startswith(prompt):
        return generated_text[len(prompt) :]
    return generated_text


def memory_snapshot(device: torch.device) -> dict[str, Any]:
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return {"mps_current_allocated_bytes": int(torch.mps.current_allocated_memory())}
    return {"device": device.type}


def write_json_or_print(payload: dict[str, Any], output_path: Path | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    if output_path is None:
        print(text, end="")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf8")
    print(text, end="")


def _generation_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    top_k = raw.get("top_k")
    return {
        "max_new_tokens": int(raw.get("max_new_tokens", 16)),
        "temperature": float(raw.get("temperature", 0.0)),
        "top_k": None if top_k in {None, 0} else int(top_k),
        "top_p": None if raw.get("top_p") in {None, 0} else float(raw["top_p"]),
        "repetition_penalty": float(raw.get("repetition_penalty", 1.0)),
        "use_cache": bool(raw.get("use_cache", False)),
        "stop_strings": list(raw.get("stop_strings", [])),
        "stop_token_ids": [int(item) for item in raw.get("stop_token_ids", [])],
    }


def _is_legacy_transformer_config(raw: dict[str, Any]) -> bool:
    return isinstance(raw.get("training"), dict) and isinstance(raw.get("optimizer"), dict)


def _required_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing mapping field: {key}")
    return value


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing string field: {key}")
    return value
