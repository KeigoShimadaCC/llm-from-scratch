from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from kgpt.checkpoint import load_checkpoint, save_checkpoint
from kgpt.git import current_git_commit
from kgpt.pretrain import (
    PretrainConfig,
    checkpoint_metadata,
    create_or_resume_run_dir,
    learning_rate_for_step,
    load_pretrain_config,
    load_resume_state,
    loss_improvement_fraction,
    perplexity,
    pretrain_config_to_dict,
    with_pretrain_overrides,
)
from kgpt.seed import seed_everything
from kgpt.token_data import Batch, TokenBatchSampler, build_tokenized_dataset_from_config
from kgpt.transformer import (
    DecoderOnlyTransformer,
    count_parameters,
    generate_tokens,
    load_tokenizer_for_config,
    resolve_device,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PHASE-04A tiny token-level pretraining.")
    parser.add_argument("--config", required=True, help="Path to tiny pretraining YAML config.")
    parser.add_argument("--max-steps", type=int, help="Override training.train_steps.")
    parser.add_argument("--run-name", help="Override run_name.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint_last.pt in the run directory.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/model wiring without training.")
    parser.add_argument(
        "--validate-resume",
        action="store_true",
        help="With --dry-run, create and reload a temporary checkpoint to validate resume metadata.",
    )
    args = parser.parse_args(argv)

    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    config = load_pretrain_config(args.config)
    config = with_pretrain_overrides(config, run_name=args.run_name, train_steps=args.max_steps)
    if args.dry_run:
        result = run_pretrain_dry_run(config=config, validate_resume=args.validate_resume)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.validate_resume:
        raise ValueError("--validate-resume requires --dry-run.")
    run_dir = create_or_resume_run_dir(config, resume=args.resume)
    result = run_pretraining(config=config, run_dir=run_dir, resume=args.resume)
    print(json.dumps(result, sort_keys=True))
    return 0


def run_pretraining(*, config: PretrainConfig, run_dir: Path, resume: bool = False) -> dict[str, Any]:
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
    parameter_count = count_parameters(model)
    if parameter_count < config.scale.min_parameters or parameter_count > config.scale.max_parameters:
        raise ValueError(
            f"{config.scale.label} model must be "
            f"{config.scale.min_parameters:,}-{config.scale.max_parameters:,} parameters, got {parameter_count:,}."
        )

    config_copy_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.jsonl"
    samples_path = run_dir / "samples.txt"
    manifest_path = run_dir / "manifest.json"
    tokenizer_info_path = run_dir / "tokenizer_info.json"
    checkpoint_last_path = run_dir / "checkpoint_last.pt"
    checkpoint_best_path = run_dir / "checkpoint_best.pt"
    if not resume:
        shutil.copyfile(config.source_path, config_copy_path)
        tokenizer_info_path.write_text(
            json.dumps(tokenizer.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf8",
        )

    train_sampler = TokenBatchSampler(
        metadata_path=config.data.metadata_path,
        split=config.data.split,
        batch_size=config.training.batch_size,
        context_length=config.model.context_length,
        seed=config.seed,
    )
    validation_sampler = TokenBatchSampler(
        metadata_path=config.data.metadata_path,
        split=config.data.validation_split,
        batch_size=config.training.batch_size,
        context_length=config.model.context_length,
        seed=config.seed + 1,
    )
    validation_batches = [validation_sampler.next_batch() for _ in range(config.training.validation_batches)]

    start_step = 0
    initial_validation_loss = evaluate_loss(model=model, batches=validation_batches, device=device)
    best_validation_loss = initial_validation_loss
    best_step = 0
    if resume:
        resume_state = load_resume_state(run_dir=run_dir, model=model, optimizer=optimizer)
        model.to(device)
        start_step = resume_state.start_step
        best_validation_loss = resume_state.best_validation_loss
        best_step = resume_state.best_step
        initial_validation_loss = resume_state.initial_validation_loss
    else:
        _record_samples(
            model=model,
            tokenizer=tokenizer,
            prompts=config.sample_prompts,
            step=0,
            samples_path=samples_path,
            config=config,
            device=device,
        )
        _save_training_checkpoint(
            path=checkpoint_best_path,
            model=model,
            optimizer=optimizer,
            config=config,
            run_dir=run_dir,
            step=0,
            parameter_count=parameter_count,
            initial_validation_loss=initial_validation_loss,
            current_validation_loss=initial_validation_loss,
            best_validation_loss=best_validation_loss,
            best_step=best_step,
            tokens_seen=0,
        )

    started = time.perf_counter()
    latest_validation_loss = initial_validation_loss if start_step == 0 else best_validation_loss
    tokens_seen = start_step * config.training.batch_size * config.training.gradient_accumulation_steps
    tokens_seen *= config.model.context_length
    starting_tokens_seen = tokens_seen
    if start_step == 0:
        _append_jsonl(
            metrics_path,
            _metrics_record(
                step=0,
                tokens_seen=0,
                train_loss=None,
                validation_loss=initial_validation_loss,
                learning_rate=0.0,
                gradient_norm=None,
                tokens_per_sec=0.0,
                device=device,
                best_validation_loss=best_validation_loss,
            ),
        )

    for step in range(start_step + 1, config.training.train_steps + 1):
        learning_rate = learning_rate_for_step(
            base_lr=config.optimizer.learning_rate,
            step=step,
            total_steps=config.training.train_steps,
            scheduler=config.scheduler,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(config.training.gradient_accumulation_steps):
            batch = train_sampler.next_batch()
            output = model(batch.inputs.to(device), targets=batch.targets.to(device))
            if output.loss is None:
                raise RuntimeError("Transformer forward did not return a training loss.")
            loss = output.loss / config.training.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += float(output.loss.detach().cpu())

        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.max_grad_norm).detach().cpu()
        )
        optimizer.step()
        tokens_seen += (
            config.training.batch_size
            * config.training.gradient_accumulation_steps
            * config.model.context_length
        )
        train_loss = accumulated_loss / config.training.gradient_accumulation_steps
        should_eval = step % config.training.eval_every == 0 or step == config.training.train_steps
        if should_eval:
            latest_validation_loss = evaluate_loss(model=model, batches=validation_batches, device=device)
            if latest_validation_loss < best_validation_loss:
                best_validation_loss = latest_validation_loss
                best_step = step
                _save_training_checkpoint(
                    path=checkpoint_best_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    run_dir=run_dir,
                    step=step,
                    parameter_count=parameter_count,
                    initial_validation_loss=initial_validation_loss,
                    current_validation_loss=latest_validation_loss,
                    best_validation_loss=best_validation_loss,
                    best_step=best_step,
                    tokens_seen=tokens_seen,
                )

        elapsed = max(time.perf_counter() - started, 1e-9)
        _append_jsonl(
            metrics_path,
            _metrics_record(
                step=step,
                tokens_seen=tokens_seen,
                train_loss=train_loss,
                validation_loss=latest_validation_loss if should_eval else None,
                learning_rate=learning_rate,
                gradient_norm=gradient_norm,
                tokens_per_sec=max(tokens_seen - starting_tokens_seen, 0) / elapsed,
                device=device,
                best_validation_loss=best_validation_loss,
            ),
        )

        if step % config.training.sample_every == 0 or step == config.training.train_steps:
            _record_samples(
                model=model,
                tokenizer=tokenizer,
                prompts=config.sample_prompts,
                step=step,
                samples_path=samples_path,
                config=config,
                device=device,
            )
        if step % config.training.checkpoint_every == 0 or step == config.training.train_steps:
            _save_training_checkpoint(
                path=checkpoint_last_path,
                model=model,
                optimizer=optimizer,
                config=config,
                run_dir=run_dir,
                step=step,
                parameter_count=parameter_count,
                initial_validation_loss=initial_validation_loss,
                current_validation_loss=latest_validation_loss,
                best_validation_loss=best_validation_loss,
                best_step=best_step,
                tokens_seen=tokens_seen,
            )

    improvement = loss_improvement_fraction(
        initial_validation_loss=initial_validation_loss,
        validation_loss=latest_validation_loss,
    )
    passed = improvement >= config.training.loss_improvement_threshold
    manifest = {
        "schema_version": 1,
        "command": "python -m train.pretrain",
        "config": pretrain_config_to_dict(config),
        "seed": config.seed,
        "device": str(device),
        "dtype": config.dtype,
        "parameter_count": parameter_count,
        "output_files": {
            "config": str(config_copy_path),
            "metrics": str(metrics_path),
            "samples": str(samples_path),
            "tokenizer_info": str(tokenizer_info_path),
            "checkpoint_last": str(checkpoint_last_path),
            "checkpoint_best": str(checkpoint_best_path),
            "manifest": str(manifest_path),
        },
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": latest_validation_loss,
        "best_validation_loss": best_validation_loss,
        "best_step": best_step,
        "loss_improvement_fraction": improvement,
        "loss_improvement_threshold": config.training.loss_improvement_threshold,
        "loss_improvement_passed": passed,
        "resume_supported": True,
        "validation_status": "pass" if passed else "fail",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf8")
    if not passed:
        raise RuntimeError(
            "PHASE-04A loss improvement gate failed: "
            f"{improvement:.3f} < {config.training.loss_improvement_threshold:.3f}."
        )
    return {
        "run_dir": str(run_dir),
        "parameter_count": parameter_count,
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": latest_validation_loss,
        "best_validation_loss": best_validation_loss,
        "best_step": best_step,
        "loss_improvement_fraction": improvement,
        "loss_improvement_passed": passed,
        "checkpoint_last": str(checkpoint_last_path),
        "checkpoint_best": str(checkpoint_best_path),
    }


def run_pretrain_dry_run(*, config: PretrainConfig, validate_resume: bool) -> dict[str, Any]:
    seed_everything(config.seed)
    data_validation = _validate_pretrain_data(config)
    tokenizer = load_tokenizer_for_config(config)
    tokenizer_validation = {
        "model_path": str(config.tokenizer.model_path),
        "fallback_training_config": str(config.tokenizer.fallback_training_config)
        if config.tokenizer.fallback_training_config
        else None,
        "tokenizer_id": tokenizer.tokenizer_id,
        "vocab_size": tokenizer.vocab_size,
        "matches_model_vocab": tokenizer.vocab_size == config.model.vocab_size,
    }
    if tokenizer.vocab_size != config.model.vocab_size:
        raise ValueError(
            f"Config vocab_size={config.model.vocab_size} does not match tokenizer vocab_size={tokenizer.vocab_size}."
        )
    device = resolve_device(config.device)
    model = DecoderOnlyTransformer(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.learning_rate)
    parameter_count = count_parameters(model)
    scale_passed = config.scale.min_parameters <= parameter_count <= config.scale.max_parameters
    if not scale_passed:
        raise ValueError(
            f"{config.scale.label} model must be "
            f"{config.scale.min_parameters:,}-{config.scale.max_parameters:,} parameters, got {parameter_count:,}."
        )
    resume_validation: dict[str, Any] | None = None
    if validate_resume:
        with tempfile.TemporaryDirectory(prefix="kgpt-resume-") as tmpdir:
            run_dir = Path(tmpdir)
            metadata = checkpoint_metadata(
                config=config,
                run_dir=run_dir,
                step=0,
                parameter_count=parameter_count,
                initial_validation_loss=1.0,
                current_validation_loss=1.0,
                best_validation_loss=1.0,
                best_step=0,
                tokens_seen=0,
                git_commit=current_git_commit(),
                created_at=datetime.now(UTC).isoformat(),
            )
            save_checkpoint(
                run_dir / "checkpoint_last.pt",
                model=model,
                optimizer=optimizer,
                metadata=metadata,
            )
            state = load_resume_state(run_dir=run_dir, model=model, optimizer=optimizer)
            checkpoint_payload = load_checkpoint(run_dir / "checkpoint_last.pt", map_location="cpu")
            checkpoint_keys = sorted(str(key) for key in checkpoint_payload["metadata"])
            resume_validation = {
                "start_step": state.start_step,
                "best_validation_loss": state.best_validation_loss,
                "best_step": state.best_step,
                "initial_validation_loss": state.initial_validation_loss,
                "checkpoint_metadata_keys": checkpoint_keys,
                "checkpoint_parameter_count": checkpoint_payload["metadata"].get("parameter_count"),
                "checkpoint_config_hash": checkpoint_payload["metadata"].get("config_hash"),
                "resume_supported": checkpoint_payload["metadata"].get("resume_supported"),
            }
    return {
        "config": str(config.source_path),
        "run_name": config.run_name,
        "model_name": config.model_name,
        "scale_label": config.scale.label,
        "parameter_count": parameter_count,
        "scale_min": config.scale.min_parameters,
        "scale_max": config.scale.max_parameters,
        "scale_passed": scale_passed,
        "device": str(device),
        "dtype": config.dtype,
        "target_steps": config.run_budget.target_steps,
        "target_tokens": config.run_budget.target_tokens,
        "data_validation": data_validation,
        "tokenizer_validation": tokenizer_validation,
        "resume_validation": resume_validation,
        "dry_run": True,
    }


def _validate_pretrain_data(config: PretrainConfig) -> dict[str, Any]:
    if not config.data.tokenized_config.is_file():
        raise FileNotFoundError(f"Missing tokenized dataset config: {config.data.tokenized_config}")
    if not config.data.metadata_path.is_file():
        build_tokenized_dataset_from_config(config.data.tokenized_config)
    if not config.data.metadata_path.is_file():
        raise FileNotFoundError(f"Missing tokenized metadata: {config.data.metadata_path}")

    metadata = json.loads(config.data.metadata_path.read_text(encoding="utf8"))
    tokenizer_payload = metadata.get("tokenizer")
    if not isinstance(tokenizer_payload, dict):
        raise ValueError(f"Tokenized metadata is missing tokenizer info: {config.data.metadata_path}")
    if int(tokenizer_payload.get("vocab_size", -1)) != config.model.vocab_size:
        raise ValueError(
            "Tokenized metadata vocab_size does not match model config: "
            f"{tokenizer_payload.get('vocab_size')} != {config.model.vocab_size}."
        )

    split_checks: dict[str, dict[str, Any]] = {}
    splits = metadata.get("splits")
    if not isinstance(splits, dict):
        raise ValueError(f"Tokenized metadata is missing splits: {config.data.metadata_path}")
    for split in (config.data.split, config.data.validation_split):
        split_payload = splits.get(split)
        if not isinstance(split_payload, dict):
            raise ValueError(f"Tokenized metadata is missing required split {split!r}: {config.data.metadata_path}")
        token_path = Path(str(split_payload.get("path", "")))
        token_count = int(split_payload.get("token_count", 0))
        if not token_path.is_file():
            raise FileNotFoundError(f"Missing token array for split {split!r}: {token_path}")
        if token_count <= config.model.context_length:
            raise ValueError(
                f"Tokenized split {split!r} has {token_count} tokens, "
                f"which is too short for context_length={config.model.context_length}."
            )
        split_checks[split] = {
            "path": str(token_path),
            "token_count": token_count,
            "record_count": int(split_payload.get("record_count", 0)),
            "sha256": split_payload.get("sha256"),
        }

    return {
        "tokenized_config": str(config.data.tokenized_config),
        "metadata_path": str(config.data.metadata_path),
        "metadata_exists": True,
        "tokenizer_vocab_size": int(tokenizer_payload["vocab_size"]),
        "splits": split_checks,
        "leakage_overlap_count": metadata.get("leakage_check", {}).get("overlap_count"),
    }


def evaluate_loss(*, model: DecoderOnlyTransformer, batches: list[Batch], device: torch.device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in batches:
            output = model(batch.inputs.to(device), targets=batch.targets.to(device))
            if output.loss is None:
                raise RuntimeError("Transformer forward did not return an evaluation loss.")
            total += float(output.loss.detach().cpu())
    return total / len(batches)


def _record_samples(
    *,
    model: DecoderOnlyTransformer,
    tokenizer: Any,
    prompts: tuple[str, ...],
    step: int,
    samples_path: Path,
    config: PretrainConfig,
    device: torch.device,
) -> None:
    with samples_path.open("a", encoding="utf8") as samples_file:
        for prompt in prompts:
            prompt_tokens = tokenizer.encode(prompt, add_bos=True)
            token_ids = generate_tokens(
                model=model,
                input_ids=prompt_tokens,
                max_new_tokens=config.sampling.max_new_tokens,
                seed=config.seed + step,
                temperature=config.sampling.temperature,
                top_k=config.sampling.top_k,
                eos_token_id=tokenizer.eos_token_id,
                device=device,
            )
            samples_file.write(
                json.dumps(
                    {
                        "step": step,
                        "prompt": prompt,
                        "generated_text": tokenizer.decode(token_ids),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def _save_training_checkpoint(
    *,
    path: Path,
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    config: PretrainConfig,
    run_dir: Path,
    step: int,
    parameter_count: int,
    initial_validation_loss: float,
    current_validation_loss: float,
    best_validation_loss: float,
    best_step: int,
    tokens_seen: int,
) -> None:
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        metadata=checkpoint_metadata(
            config=config,
            run_dir=run_dir,
            step=step,
            parameter_count=parameter_count,
            initial_validation_loss=initial_validation_loss,
            current_validation_loss=current_validation_loss,
            best_validation_loss=best_validation_loss,
            best_step=best_step,
            tokens_seen=tokens_seen,
            git_commit=current_git_commit(),
            created_at=datetime.now(UTC).isoformat(),
        ),
    )


def _metrics_record(
    *,
    step: int,
    tokens_seen: int,
    train_loss: float | None,
    validation_loss: float | None,
    learning_rate: float,
    gradient_norm: float | None,
    tokens_per_sec: float,
    device: torch.device,
    best_validation_loss: float,
) -> dict[str, Any]:
    return {
        "step": step,
        "tokens_seen": tokens_seen,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "perplexity": perplexity(validation_loss) if validation_loss is not None else None,
        "best_validation_loss": best_validation_loss,
        "learning_rate": learning_rate,
        "gradient_norm": gradient_norm,
        "tokens_per_sec": tokens_per_sec,
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
