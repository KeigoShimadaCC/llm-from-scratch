from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from kgpt.checkpoint import save_checkpoint
from kgpt.config import TrainingConfig, config_to_dict, load_training_config
from kgpt.git import current_git_commit
from kgpt.seed import seed_everything


class DummyTokenModel(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(context_length * hidden_size, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        return self.proj(embedded.flatten(start_dim=1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic fake-token training smoke test.")
    parser.add_argument("--config", required=True, help="Path to YAML training config.")
    parser.add_argument("--run-name", help="Override run_name from config.")
    args = parser.parse_args(argv)

    config = load_training_config(args.config)
    if args.run_name:
        config = _replace_run_name(config, args.run_name)

    run_dir = create_run_dir(config)
    result = run_dummy_training(config, run_dir)
    print(json.dumps({"run_dir": str(run_dir), "final_loss": result["final_loss"]}, sort_keys=True))
    return 0


def create_run_dir(config: TrainingConfig) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", config.run_name).strip("-") or "run"
    run_dir = config.output_dir / f"{timestamp}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_dummy_training(config: TrainingConfig, run_dir: Path) -> dict[str, Any]:
    seed_everything(config.seed)
    device = _resolve_device(config.device)
    model = DummyTokenModel(config.vocab_size, config.context_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.learning_rate)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    metrics_path = run_dir / "metrics.jsonl"
    samples_path = run_dir / "samples.txt"
    config_copy_path = run_dir / "config.yaml"
    manifest_path = run_dir / "manifest.json"
    shutil.copyfile(config.source_path, config_copy_path)

    started = time.perf_counter()
    last_metrics: dict[str, Any] = {}
    for step in range(1, config.train_steps + 1):
        tokens = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(config.batch_size, config.context_length + 1),
            generator=generator,
            dtype=torch.long,
        )
        inputs = tokens[:, : config.context_length].to(device)
        targets = tokens[:, -1].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).detach().cpu())
        optimizer.step()

        elapsed = max(time.perf_counter() - started, 1e-9)
        tokens_seen = step * config.batch_size * config.context_length
        last_metrics = {
            "step": step,
            "tokens_seen": tokens_seen,
            "train_loss": float(loss.detach().cpu()),
            "validation_loss": None,
            "perplexity": float(math.exp(min(float(loss.detach().cpu()), 20.0))),
            "learning_rate": config.optimizer.learning_rate,
            "gradient_norm": grad_norm,
            "tokens_per_sec": tokens_seen / elapsed,
            "memory_usage": _memory_usage(device),
            "sample_path": str(samples_path),
        }
        with metrics_path.open("a", encoding="utf8") as metrics_file:
            metrics_file.write(json.dumps(last_metrics, sort_keys=True) + "\n")

    sample_tokens = torch.randint(0, config.vocab_size, (32,), generator=generator).tolist()
    samples_path.write_text(" ".join(str(token) for token in sample_tokens) + "\n", encoding="utf8")
    checkpoint_path = run_dir / "checkpoint_last.pt"
    metadata = {
        "config_hash": config.config_hash,
        "config_path": str(config_copy_path),
        "model_name": config.model_name,
        "step": config.train_steps,
        "seed": config.seed,
        "git_commit": current_git_commit(),
        "created_at": datetime.now(UTC).isoformat(),
        "metrics": last_metrics,
        "tokenizer_id": config.tokenizer_id,
    }
    save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, metadata=metadata)
    manifest = {
        "schema_version": 1,
        "command": "python -m train.dummy",
        "config": config_to_dict(config),
        "seed": config.seed,
        "device": str(device),
        "output_files": {
            "config": str(config_copy_path),
            "metrics": str(metrics_path),
            "samples": str(samples_path),
            "checkpoint": str(checkpoint_path),
        },
        "validation_status": "pass",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf8")
    return {"run_dir": run_dir, "final_loss": last_metrics["train_loss"]}


def _resolve_device(requested: str) -> torch.device:
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _memory_usage(device: torch.device) -> dict[str, Any]:
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return {"mps_current_allocated_bytes": int(torch.mps.current_allocated_memory())}
    return {"device": device.type}


def _replace_run_name(config: TrainingConfig, run_name: str) -> TrainingConfig:
    return TrainingConfig(
        run_name=run_name,
        seed=config.seed,
        device=config.device,
        dtype=config.dtype,
        model_name=config.model_name,
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        batch_size=config.batch_size,
        train_steps=config.train_steps,
        optimizer=config.optimizer,
        output_dir=config.output_dir,
        checkpoint_every=config.checkpoint_every,
        eval_every=config.eval_every,
        sample_every=config.sample_every,
        tokenizer_id=config.tokenizer_id,
        source_path=config.source_path,
        config_hash=config.config_hash,
    )


if __name__ == "__main__":
    raise SystemExit(main())
