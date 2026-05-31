from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from kgpt.checkpoint import save_checkpoint
from kgpt.git import current_git_commit
from kgpt.seed import seed_everything
from kgpt.token_data import TokenBatchSampler, build_tokenized_dataset_from_config
from kgpt.transformer import (
    DecoderOnlyTransformer,
    count_parameters,
    load_tokenizer_for_config,
    load_transformer_experiment_config,
    resolve_device,
    transformer_config_to_dict,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a PHASE-03A micro Transformer smoke run.")
    parser.add_argument("--config", required=True, help="Path to transformer smoke YAML config.")
    parser.add_argument("--max-steps", type=int, help="Override train_steps from config.")
    args = parser.parse_args(argv)

    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    config = load_transformer_experiment_config(args.config)
    run_dir = create_run_dir(config.run_name, config.training.output_dir)
    result = run_transformer_smoke_training(config=config, run_dir=run_dir, max_steps=args.max_steps)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "initial_loss": result["initial_loss"],
                "final_loss": result["final_loss"],
                "loss_improved": result["loss_improved"],
                "checkpoint": str(result["checkpoint"]),
            },
            sort_keys=True,
        )
    )
    return 0


def create_run_dir(run_name: str, output_dir: Path) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", run_name).strip("-") or "run"
    run_dir = output_dir / slug
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_transformer_smoke_training(
    *,
    config: Any,
    run_dir: Path,
    max_steps: int | None = None,
) -> dict[str, Any]:
    seed_everything(config.seed)
    build_tokenized_dataset_from_config(config.data.tokenized_config)
    tokenizer = load_tokenizer_for_config(config)
    if tokenizer.vocab_size != config.model.vocab_size:
        raise ValueError(
            f"Config vocab_size={config.model.vocab_size} does not match tokenizer vocab_size={tokenizer.vocab_size}."
        )

    device = resolve_device(config.device)
    model = DecoderOnlyTransformer(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.learning_rate)
    sampler = TokenBatchSampler(
        metadata_path=config.data.metadata_path,
        split=config.data.split,
        batch_size=config.training.batch_size,
        context_length=config.model.context_length,
        seed=config.seed,
    )
    fixed_batch = sampler.next_batch()
    train_steps = max_steps or config.training.train_steps

    config_copy_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.jsonl"
    manifest_path = run_dir / "manifest.json"
    checkpoint_path = run_dir / "checkpoint_last.pt"
    shutil.copyfile(config.source_path, config_copy_path)

    started = time.perf_counter()
    initial_loss = evaluate_batch_loss(model=model, batch=fixed_batch, device=device)
    _append_jsonl(
        metrics_path,
        _metrics_record(
            step=0,
            loss=initial_loss,
            tokens_seen=0,
            learning_rate=config.optimizer.learning_rate,
            gradient_norm=None,
            tokens_per_sec=0.0,
            device=device,
        ),
    )
    last_loss = initial_loss
    last_grad_norm = None
    for step in range(1, train_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        inputs = fixed_batch.inputs.to(device)
        targets = fixed_batch.targets.to(device)
        output = model(inputs, targets=targets)
        if output.loss is None:
            raise RuntimeError("Transformer forward did not return a training loss.")
        output.loss.backward()
        last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).detach().cpu())
        optimizer.step()
        last_loss = float(output.loss.detach().cpu())
        if step == train_steps or step % config.training.eval_every == 0:
            last_loss = evaluate_batch_loss(model=model, batch=fixed_batch, device=device)
        elapsed = max(time.perf_counter() - started, 1e-9)
        tokens_seen = step * config.training.batch_size * config.model.context_length
        _append_jsonl(
            metrics_path,
            _metrics_record(
                step=step,
                loss=last_loss,
                tokens_seen=tokens_seen,
                learning_rate=config.optimizer.learning_rate,
                gradient_norm=last_grad_norm,
                tokens_per_sec=tokens_seen / elapsed,
                device=device,
            ),
        )

    final_loss = evaluate_batch_loss(model=model, batch=fixed_batch, device=device)
    loss_improved = final_loss < initial_loss
    parameter_count = count_parameters(model)
    metadata = {
        "config_hash": config.config_hash,
        "config_path": str(config_copy_path),
        "model_name": config.model_name,
        "model_config": transformer_config_to_dict(config)["model"],
        "parameter_count": parameter_count,
        "step": train_steps,
        "seed": config.seed,
        "git_commit": current_git_commit(),
        "created_at": datetime.now(UTC).isoformat(),
        "device": str(device),
        "dtype": config.dtype,
        "tokenizer": {
            "tokenizer_id": tokenizer.tokenizer_id,
            "vocab_size": tokenizer.vocab_size,
            "model_path": str(config.tokenizer.model_path),
        },
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_improved": loss_improved,
        "training_batch": {
            "split": config.data.split,
            "batch_size": config.training.batch_size,
            "context_length": config.model.context_length,
            "metadata_path": str(config.data.metadata_path),
        },
    }
    save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, metadata=metadata)
    manifest = {
        "schema_version": 1,
        "command": "python -m train.transformer_smoke",
        "config": transformer_config_to_dict(config),
        "seed": config.seed,
        "device": str(device),
        "dtype": config.dtype,
        "parameter_count": parameter_count,
        "output_files": {
            "config": str(config_copy_path),
            "metrics": str(metrics_path),
            "manifest": str(manifest_path),
            "checkpoint": str(checkpoint_path),
        },
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_improved": loss_improved,
        "validation_status": "pass" if loss_improved else "fail",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf8")
    return {
        "run_dir": run_dir,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_improved": loss_improved,
        "checkpoint": checkpoint_path,
    }


def evaluate_batch_loss(*, model: DecoderOnlyTransformer, batch: Any, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        output = model(batch.inputs.to(device), targets=batch.targets.to(device))
    if output.loss is None:
        raise RuntimeError("Transformer forward did not return an evaluation loss.")
    return float(output.loss.detach().cpu())


def _metrics_record(
    *,
    step: int,
    loss: float,
    tokens_seen: int,
    learning_rate: float,
    gradient_norm: float | None,
    tokens_per_sec: float,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "step": step,
        "train_loss": loss,
        "learning_rate": learning_rate,
        "gradient_norm": gradient_norm,
        "tokens_seen": tokens_seen,
        "tokens_per_sec": tokens_per_sec,
        "device": str(device),
        "memory_usage": _memory_usage(device),
    }


def _memory_usage(device: torch.device) -> dict[str, Any]:
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return {"mps_current_allocated_bytes": int(torch.mps.current_allocated_memory())}
    return {"device": device.type}


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf8") as metrics_file:
        metrics_file.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
