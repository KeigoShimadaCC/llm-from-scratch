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
import torch.nn as nn

from kgpt.char_tokenizer import CharacterTokenizer
from kgpt.checkpoint import save_checkpoint
from kgpt.git import current_git_commit
from kgpt.micro_char import (
    CharContextLanguageModel,
    MicroCharConfig,
    build_next_char_dataset,
    config_to_dict,
    count_parameters,
    evaluate_loss,
    generate_text,
    load_micro_char_config,
    perplexity,
    split_dataset,
    with_overrides,
)
from kgpt.seed import seed_everything


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the PHASE-01A micro character language model.")
    parser.add_argument("--config", required=True, help="Path to YAML micro character config.")
    parser.add_argument("--run-name", help="Override run_name from config.")
    parser.add_argument("--max-steps", type=int, help="Override train_steps from config.")
    args = parser.parse_args(argv)

    config = load_micro_char_config(args.config)
    if args.run_name or args.max_steps is not None:
        if args.max_steps is not None and args.max_steps <= 0:
            raise ValueError("--max-steps must be positive.")
        config = with_overrides(config, run_name=args.run_name, train_steps=args.max_steps)

    run_dir = create_run_dir(config)
    result = run_micro_char_training(config, run_dir)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "initial_train_loss": result["initial_train_loss"],
                "final_train_loss": result["final_train_loss"],
                "overfit_passed": result["overfit_passed"],
            },
            sort_keys=True,
        )
    )
    return 0


def create_run_dir(config: MicroCharConfig) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", config.run_name).strip("-") or "run"
    run_dir = config.output_dir / slug
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_micro_char_training(config: MicroCharConfig, run_dir: Path) -> dict[str, Any]:
    seed_everything(config.seed)
    device = _resolve_device(config.device)
    tokenizer = CharacterTokenizer.build([config.data.text], tokenizer_id=config.tokenizer_id)
    token_ids = tokenizer.encode(config.data.text)
    inputs, targets = build_next_char_dataset(token_ids, context_length=config.context_length)
    train_inputs, train_targets, validation_inputs, validation_targets = split_dataset(
        inputs,
        targets,
        train_fraction=config.data.train_fraction,
    )

    model = CharContextLanguageModel(
        vocab_size=tokenizer.vocab_size,
        context_length=config.context_length,
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.learning_rate)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    metrics_path = run_dir / "metrics.jsonl"
    samples_path = run_dir / "samples.txt"
    config_copy_path = run_dir / "config.yaml"
    tokenizer_path = run_dir / "tokenizer.json"
    manifest_path = run_dir / "manifest.json"
    checkpoint_path = run_dir / "checkpoint_last.pt"
    shutil.copyfile(config.source_path, config_copy_path)
    tokenizer.save(tokenizer_path)

    parameter_count = count_parameters(model)
    initial_train_loss = evaluate_loss(model, train_inputs, train_targets, device=device)
    initial_validation_loss = evaluate_loss(model, validation_inputs, validation_targets, device=device)
    started = time.perf_counter()
    last_metrics = _metrics_record(
        step=0,
        tokens_seen=0,
        train_loss=initial_train_loss,
        validation_loss=initial_validation_loss,
        learning_rate=config.optimizer.learning_rate,
        gradient_norm=None,
        tokens_per_sec=0.0,
        device=device,
        sample_path=samples_path,
    )
    _append_jsonl(metrics_path, last_metrics)
    _append_sample(
        samples_path,
        step=0,
        mode="greedy",
        text=generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=config.generation.prompt,
            max_new_tokens=config.generation.max_new_tokens,
            seed=config.seed,
            temperature=0.0,
            greedy=True,
            device=device,
        ),
    )

    for step in range(1, config.train_steps + 1):
        batch_inputs, batch_targets = _batch(
            train_inputs,
            train_targets,
            batch_size=config.batch_size,
            generator=generator,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_inputs.to(device))
        loss = criterion(logits, batch_targets.to(device))
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).detach().cpu())
        optimizer.step()

        should_eval = step == config.train_steps or step % config.eval_every == 0
        train_loss = (
            evaluate_loss(model, train_inputs, train_targets, device=device)
            if should_eval
            else float(loss.detach().cpu())
        )
        validation_loss = (
            evaluate_loss(model, validation_inputs, validation_targets, device=device) if should_eval else None
        )
        elapsed = max(time.perf_counter() - started, 1e-9)
        tokens_seen = step * config.batch_size * config.context_length
        last_metrics = _metrics_record(
            step=step,
            tokens_seen=tokens_seen,
            train_loss=train_loss,
            validation_loss=validation_loss,
            learning_rate=config.optimizer.learning_rate,
            gradient_norm=grad_norm,
            tokens_per_sec=tokens_seen / elapsed,
            device=device,
            sample_path=samples_path,
        )
        _append_jsonl(metrics_path, last_metrics)

        if step % config.sample_every == 0 or step == config.train_steps:
            _append_sample(
                samples_path,
                step=step,
                mode=f"sample_temperature_{config.generation.temperature}",
                text=generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=config.generation.prompt,
                    max_new_tokens=config.generation.max_new_tokens,
                    seed=config.seed + step,
                    temperature=config.generation.temperature,
                    greedy=False,
                    device=device,
                ),
            )

    final_train_loss = evaluate_loss(model, train_inputs, train_targets, device=device)
    final_validation_loss = evaluate_loss(model, validation_inputs, validation_targets, device=device)
    last_metrics = {
        **last_metrics,
        "train_loss": final_train_loss,
        "validation_loss": final_validation_loss,
        "perplexity": perplexity(final_train_loss),
    }
    overfit_passed = final_train_loss < config.overfit_threshold
    _append_sample(
        samples_path,
        step=config.train_steps,
        mode="greedy_final",
        text=generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=config.generation.prompt,
            max_new_tokens=config.generation.max_new_tokens,
            seed=config.seed,
            temperature=0.0,
            greedy=True,
            device=device,
        ),
    )

    metadata = {
        "config_hash": config.config_hash,
        "config_path": str(config_copy_path),
        "model_name": config.model_name,
        "model_config": {
            "vocab_size": tokenizer.vocab_size,
            "context_length": config.context_length,
            "embedding_dim": config.model.embedding_dim,
            "hidden_dim": config.model.hidden_dim,
            "dropout": config.model.dropout,
        },
        "parameter_count": parameter_count,
        "step": config.train_steps,
        "seed": config.seed,
        "git_commit": current_git_commit(),
        "created_at": datetime.now(UTC).isoformat(),
        "metrics": last_metrics,
        "initial_metrics": {
            "train_loss": initial_train_loss,
            "validation_loss": initial_validation_loss,
        },
        "tokenizer_id": config.tokenizer_id,
        "tokenizer": tokenizer.to_dict(),
        "corpus": {
            "name": config.data.name,
            "source": config.data.source,
            "license_note": config.data.license_note,
            "characters": len(config.data.text),
            "train_examples": int(train_inputs.shape[0]),
            "validation_examples": int(validation_inputs.shape[0]),
        },
        "overfit_threshold": config.overfit_threshold,
        "overfit_passed": overfit_passed,
    }
    save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, metadata=metadata)
    manifest = {
        "schema_version": 1,
        "command": "python -m train.micro_char",
        "config": config_to_dict(config),
        "seed": config.seed,
        "device": str(device),
        "parameter_count": parameter_count,
        "output_files": {
            "config": str(config_copy_path),
            "metrics": str(metrics_path),
            "samples": str(samples_path),
            "tokenizer": str(tokenizer_path),
            "checkpoint": str(checkpoint_path),
        },
        "initial_metrics": {
            "train_loss": initial_train_loss,
            "validation_loss": initial_validation_loss,
        },
        "final_metrics": {
            "train_loss": final_train_loss,
            "validation_loss": final_validation_loss,
            "perplexity": perplexity(final_train_loss),
        },
        "overfit_threshold": config.overfit_threshold,
        "overfit_passed": overfit_passed,
        "validation_status": "pass" if overfit_passed else "fail",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf8")
    return {
        "run_dir": run_dir,
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "final_validation_loss": final_validation_loss,
        "overfit_passed": overfit_passed,
    }


def _batch(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size >= inputs.shape[0]:
        return inputs, targets
    indices = torch.randint(0, inputs.shape[0], (batch_size,), generator=generator)
    return inputs[indices], targets[indices]


def _metrics_record(
    *,
    step: int,
    tokens_seen: int,
    train_loss: float,
    validation_loss: float | None,
    learning_rate: float,
    gradient_norm: float | None,
    tokens_per_sec: float,
    device: torch.device,
    sample_path: Path,
) -> dict[str, Any]:
    return {
        "step": step,
        "tokens_seen": tokens_seen,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "perplexity": perplexity(train_loss),
        "learning_rate": learning_rate,
        "gradient_norm": gradient_norm,
        "tokens_per_sec": tokens_per_sec,
        "memory_usage": _memory_usage(device),
        "sample_path": str(sample_path),
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf8") as metrics_file:
        metrics_file.write(json.dumps(payload, sort_keys=True) + "\n")


def _append_sample(path: Path, *, step: int, mode: str, text: str) -> None:
    with path.open("a", encoding="utf8") as samples_file:
        samples_file.write(f"step={step} mode={mode}\n{text}\n\n")


def _resolve_device(requested: str) -> torch.device:
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _memory_usage(device: torch.device) -> dict[str, Any]:
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return {"mps_current_allocated_bytes": int(torch.mps.current_allocated_memory())}
    return {"device": device.type}


if __name__ == "__main__":
    raise SystemExit(main())
