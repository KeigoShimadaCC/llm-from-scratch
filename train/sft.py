from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from kgpt.checkpoint import load_checkpoint, save_checkpoint
from kgpt.git import current_git_commit
from kgpt.seed import seed_everything
from kgpt.sft import (
    SFTConfig,
    SFTExample,
    build_sft_examples,
    load_base_pretrain_config,
    load_sft_config,
    sft_config_to_dict,
    split_instruction_records,
    with_sft_overrides,
    write_instruction_manifest,
)
from kgpt.transformer import DecoderOnlyTransformer, generate_tokens, load_tokenizer_for_config, resolve_device
from train.pretrain import create_or_resume_run_dir, run_pretraining


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PHASE-06A supervised instruction tuning.")
    parser.add_argument("--config", required=True, help="Path to SFT YAML config.")
    parser.add_argument("--max-steps", type=int, help="Override SFT train_steps.")
    parser.add_argument("--run-name", help="Override SFT run_name.")
    args = parser.parse_args(argv)
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")

    config = load_sft_config(args.config)
    config = with_sft_overrides(config, run_name=args.run_name, train_steps=args.max_steps)
    run_dir = create_sft_run_dir(config)
    result = run_sft(config=config, run_dir=run_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


def create_sft_run_dir(config: SFTConfig) -> Path:
    run_dir = config.training.output_dir / config.run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_sft(*, config: SFTConfig, run_dir: Path) -> dict[str, Any]:
    seed_everything(config.seed)
    base_config = load_base_pretrain_config(config)
    if not config.base.checkpoint.is_file():
        if not config.base.bootstrap_if_missing:
            raise FileNotFoundError(f"Base checkpoint not found: {config.base.checkpoint}")
        bootstrap_config = load_base_pretrain_config(config)
        bootstrap_config = bootstrap_config.__class__(
            **{
                **bootstrap_config.__dict__,
                "run_name": config.base.bootstrap_run_name,
                "training": bootstrap_config.training.__class__(
                    **{**bootstrap_config.training.__dict__, "train_steps": config.base.bootstrap_max_steps}
                ),
            }
        )
        bootstrap_run_dir = create_or_resume_run_dir(bootstrap_config, resume=False)
        run_pretraining(config=bootstrap_config, run_dir=bootstrap_run_dir, resume=False)

    tokenizer = load_tokenizer_for_config(base_config)
    splits = split_instruction_records(
        config.dataset.records,
        validation_fraction=config.dataset.validation_fraction,
        seed=config.dataset.split_seed,
    )
    write_instruction_manifest(config, splits)
    train_examples = build_sft_examples(
        records=splits["train"],
        tokenizer=tokenizer,
        template=config.prompt_template,
        context_length=base_config.model.context_length,
        response_only_loss=config.dataset.response_only_loss,
    )
    validation_examples = build_sft_examples(
        records=splits["validation"],
        tokenizer=tokenizer,
        template=config.prompt_template,
        context_length=base_config.model.context_length,
        response_only_loss=config.dataset.response_only_loss,
    )

    device = resolve_device(config.device)
    model = DecoderOnlyTransformer(base_config.model)
    load_checkpoint(config.base.checkpoint, model=model, map_location="cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)

    config_copy_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.jsonl"
    samples_path = run_dir / "samples.txt"
    manifest_path = run_dir / "manifest.json"
    checkpoint_path = run_dir / "checkpoint_last.pt"
    shutil.copyfile(config.source_path, config_copy_path)

    initial_validation_loss = evaluate_sft_loss(model=model, examples=validation_examples, device=device)
    _append_jsonl(
        metrics_path,
        _metrics_record(
            step=0,
            train_loss=None,
            validation_loss=initial_validation_loss,
            tokens_seen=0,
            tokens_per_sec=0.0,
            gradient_norm=None,
        ),
    )
    _record_samples(
        model=model,
        tokenizer=tokenizer,
        examples=validation_examples,
        samples_path=samples_path,
        step=0,
        config=config,
        device=device,
    )

    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    started = time.perf_counter()
    latest_validation_loss = initial_validation_loss
    latest_train_loss = initial_validation_loss
    for step in range(1, config.training.train_steps + 1):
        batch = _sample_examples(train_examples, batch_size=config.training.batch_size, generator=generator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _batch_loss(model=model, examples=batch, device=device)
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.max_grad_norm).detach().cpu()
        )
        optimizer.step()
        latest_train_loss = float(loss.detach().cpu())
        should_eval = step % config.training.eval_every == 0 or step == config.training.train_steps
        if should_eval:
            latest_validation_loss = evaluate_sft_loss(model=model, examples=validation_examples, device=device)
        elapsed = max(time.perf_counter() - started, 1e-9)
        tokens_seen = step * config.training.batch_size * base_config.model.context_length
        _append_jsonl(
            metrics_path,
            _metrics_record(
                step=step,
                train_loss=latest_train_loss,
                validation_loss=latest_validation_loss if should_eval else None,
                tokens_seen=tokens_seen,
                tokens_per_sec=tokens_seen / elapsed,
                gradient_norm=gradient_norm,
            ),
        )
        if step % config.training.sample_every == 0 or step == config.training.train_steps:
            _record_samples(
                model=model,
                tokenizer=tokenizer,
                examples=validation_examples,
                samples_path=samples_path,
                step=step,
                config=config,
                device=device,
            )

    metadata = {
        "config_hash": config.config_hash,
        "config_path": str(config_copy_path),
        "base_config": str(config.base.config),
        "base_checkpoint": str(config.base.checkpoint),
        "prompt_template_version": config.prompt_template.version,
        "response_only_loss": config.dataset.response_only_loss,
        "dataset_manifest": str(config.dataset.manifest_path),
        "step": config.training.train_steps,
        "seed": config.seed,
        "git_commit": current_git_commit(),
        "created_at": datetime.now(UTC).isoformat(),
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": latest_validation_loss,
        "sft_config": sft_config_to_dict(config),
    }
    save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, metadata=metadata)
    manifest = {
        "schema_version": 1,
        "command": "python -m train.sft",
        "config": sft_config_to_dict(config),
        "output_files": {
            "config": str(config_copy_path),
            "metrics": str(metrics_path),
            "samples": str(samples_path),
            "checkpoint_last": str(checkpoint_path),
            "manifest": str(manifest_path),
        },
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": latest_validation_loss,
        "loss_improved": latest_validation_loss < initial_validation_loss,
        "prompt_template_version": config.prompt_template.version,
        "response_only_loss": config.dataset.response_only_loss,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf8")
    return {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": latest_validation_loss,
        "loss_improved": latest_validation_loss < initial_validation_loss,
        "prompt_template_version": config.prompt_template.version,
    }


def evaluate_sft_loss(*, model: DecoderOnlyTransformer, examples: list[SFTExample], device: torch.device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for example in examples:
            total += float(_batch_loss(model=model, examples=[example], device=device).detach().cpu())
    return total / len(examples)


def _batch_loss(*, model: DecoderOnlyTransformer, examples: list[SFTExample], device: torch.device) -> torch.Tensor:
    inputs = torch.stack([example.input_ids for example in examples]).to(device)
    targets = torch.stack([example.targets for example in examples]).to(device)
    output = model(inputs, targets=targets)
    if output.loss is None:
        raise RuntimeError("SFT forward did not return a loss.")
    return output.loss


def _sample_examples(
    examples: list[SFTExample],
    *,
    batch_size: int,
    generator: torch.Generator,
) -> list[SFTExample]:
    indices = torch.randint(0, len(examples), (batch_size,), generator=generator)
    return [examples[int(index)] for index in indices]


def _record_samples(
    *,
    model: DecoderOnlyTransformer,
    tokenizer: Any,
    examples: list[SFTExample],
    samples_path: Path,
    step: int,
    config: SFTConfig,
    device: torch.device,
) -> None:
    with samples_path.open("a", encoding="utf8") as samples_file:
        for record in config.dataset.records[: min(4, len(config.dataset.records))]:
            prompt = config.prompt_template.prompt_pattern.format(instruction=record.instruction)
            token_ids = generate_tokens(
                model=model,
                input_ids=tokenizer.encode(prompt, add_bos=True),
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
                        "instruction": record.instruction,
                        "expected_response": record.response,
                        "generated_text": tokenizer.decode(token_ids),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def _metrics_record(
    *,
    step: int,
    train_loss: float | None,
    validation_loss: float | None,
    tokens_seen: int,
    tokens_per_sec: float,
    gradient_norm: float | None,
) -> dict[str, Any]:
    return {
        "step": step,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "tokens_seen": tokens_seen,
        "tokens_per_sec": tokens_per_sec,
        "gradient_norm": gradient_norm,
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf8") as metrics_file:
        metrics_file.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
