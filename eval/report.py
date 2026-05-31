from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from kgpt.checkpoint import load_checkpoint
from kgpt.pretrain import load_pretrain_config, loss_improvement_fraction
from kgpt.seed import seed_everything
from kgpt.transformer import DecoderOnlyTransformer, generate_tokens, load_tokenizer_for_config, resolve_device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a PHASE-04A tiny pretraining report.")
    parser.add_argument("--config", required=True, help="Path to fixed-prompt eval YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_last.pt.")
    parser.add_argument("--output", required=True, help="Markdown report output path.")
    args = parser.parse_args(argv)

    report = generate_report(
        eval_config_path=Path(args.config),
        checkpoint_path=Path(args.checkpoint),
        output_path=Path(args.output),
    )
    print(json.dumps(report, sort_keys=True))
    return 0


def generate_report(*, eval_config_path: Path, checkpoint_path: Path, output_path: Path) -> dict[str, Any]:
    eval_config = _load_eval_config(eval_config_path)
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    metadata = payload["metadata"]
    run_dir = checkpoint_path.parent
    pretrain_config = load_pretrain_config(run_dir / "config.yaml")
    tokenizer = load_tokenizer_for_config(pretrain_config)
    model = DecoderOnlyTransformer(pretrain_config.model)
    model.load_state_dict(payload["model_state"])
    device = resolve_device(pretrain_config.device)
    model.to(device)
    seed_everything(pretrain_config.seed)

    metrics = _read_jsonl(run_dir / "metrics.jsonl")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf8"))
    data_manifest = json.loads(Path("docs/phase04a_data_manifest.json").read_text(encoding="utf8"))
    tokenized_metadata = json.loads(Path(manifest["config"]["data"]["metadata_path"]).read_text(encoding="utf8"))
    sample_history = _read_jsonl(run_dir / "samples.txt")
    prompts = eval_config["fixed_prompts"]
    samples = _generate_fixed_prompt_samples(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        generation_config=eval_config["generation"],
        seed=pretrain_config.seed,
        device=device,
    )
    initial_validation_loss = float(manifest["initial_validation_loss"])
    final_validation_loss = float(manifest["final_validation_loss"])
    improvement = loss_improvement_fraction(
        initial_validation_loss=initial_validation_loss,
        validation_loss=final_validation_loss,
    )
    report_text = _render_markdown_report(
        eval_config=eval_config,
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        manifest=manifest,
        data_manifest=data_manifest,
        tokenized_metadata=tokenized_metadata,
        metrics=metrics,
        sample_history=sample_history,
        generated_samples=samples,
        improvement=improvement,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf8")
    return {
        "output": str(output_path),
        "checkpoint": str(checkpoint_path),
        "parameter_count": metadata["parameter_count"],
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": final_validation_loss,
        "loss_improvement_fraction": improvement,
        "loss_improvement_passed": manifest["loss_improvement_passed"],
        "sample_count": len(samples),
    }


def _load_eval_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Eval config must be a mapping: {path}")
    prompts = raw.get("fixed_prompts")
    generation = raw.get("generation")
    if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
        raise ValueError("fixed_prompts must be a list of strings.")
    if not isinstance(generation, dict):
        raise ValueError("generation must be a mapping.")
    return {
        "title": str(raw.get("title", "PHASE-04A tiny pretraining report")),
        "run_name": str(raw.get("run_name", "")),
        "fixed_prompts": prompts,
        "generation": {
            "max_new_tokens": int(generation.get("max_new_tokens", 32)),
            "temperature": float(generation.get("temperature", 0.0)),
            "top_k": None if generation.get("top_k") in {None, 0} else int(generation["top_k"]),
        },
        "next_scale_recommendation": str(raw.get("next_scale_recommendation", "")),
    }


def _generate_fixed_prompt_samples(
    *,
    model: DecoderOnlyTransformer,
    tokenizer: Any,
    prompts: list[str],
    generation_config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, prompt in enumerate(prompts):
        token_ids = generate_tokens(
            model=model,
            input_ids=tokenizer.encode(prompt, add_bos=True),
            max_new_tokens=generation_config["max_new_tokens"],
            seed=seed + index,
            temperature=generation_config["temperature"],
            top_k=generation_config["top_k"],
            eos_token_id=tokenizer.eos_token_id,
            device=device,
        )
        rows.append({"prompt": prompt, "generated_text": _clean_text(tokenizer.decode(token_ids))})
    return rows


def _render_markdown_report(
    *,
    eval_config: dict[str, Any],
    checkpoint_path: Path,
    metadata: dict[str, Any],
    manifest: dict[str, Any],
    data_manifest: dict[str, Any],
    tokenized_metadata: dict[str, Any],
    metrics: list[dict[str, Any]],
    sample_history: list[dict[str, Any]],
    generated_samples: list[dict[str, str]],
    improvement: float,
) -> str:
    config = manifest["config"]
    model_config = config["model"]
    training_config = config["training"]
    run_budget = config["run_budget"]
    tokenizer_info = tokenized_metadata["tokenizer"]
    eval_points = [row for row in metrics if row.get("validation_loss") is not None]
    curve = "\n".join(
        f"- step {row['step']}: validation_loss={row['validation_loss']:.4f}, "
        f"perplexity={row['perplexity']:.2f}"
        for row in eval_points
    )
    sample_block = "\n".join(
        f"### Prompt: `{sample['prompt']}`\n\n```text\n{sample['generated_text']}\n```" for sample in generated_samples
    )
    progression_block = _sample_progression_block(sample_history)
    return f"""# {eval_config["title"]}

## Summary

- Checkpoint: `{checkpoint_path}`
- Model: `{metadata["model_name"]}`
- Parameter count: {metadata["parameter_count"]:,}
- Initial validation loss: {manifest["initial_validation_loss"]:.4f}
- Final validation loss: {manifest["final_validation_loss"]:.4f}
- Best validation loss: {manifest["best_validation_loss"]:.4f} at step {manifest["best_step"]}
- Loss improvement: {improvement:.2%}
- Predeclared threshold: {manifest["loss_improvement_threshold"]:.2%}
- Gate: {"pass" if manifest["loss_improvement_passed"] else "fail"}

## Dataset Source And License

- Source: {data_manifest["source_name"]}
- License: {data_manifest["license"]}
- Language mix: {data_manifest["language_mix"]}
- Split method: {data_manifest["split_method"]}
- Dedup strategy: {data_manifest["dedup_strategy"]}
- Leakage check: {data_manifest["leakage_check"]["overlap_count"]} overlapping normalized text hashes.

## Tokenizer

- Tokenizer id: {tokenizer_info["tokenizer_id"]}
- Algorithm: {tokenizer_info["algorithm"]}
- Vocabulary size: {tokenizer_info["vocab_size"]}
- Byte fallback: {tokenizer_info["byte_fallback"]}

## Model Config

- Context length: {model_config["context_length"]}
- Embedding dimension: {model_config["embedding_dim"]}
- Layers: {model_config["num_layers"]}
- Heads: {model_config["num_heads"]}
- MLP hidden dimension: {model_config["mlp_hidden_dim"]}
- Tied embeddings: {model_config["tie_embeddings"]}

## Run Budget And Hyperparameters

- Target steps: {run_budget["target_steps"]}
- Target tokens: {run_budget["target_tokens"]}
- Target wall clock: {run_budget["target_wall_clock_minutes"]} minutes
- Hardware/device: {run_budget["hardware"]} / `{manifest["device"]}`
- Optimizer: {config["optimizer"]["name"]}, lr={config["optimizer"]["learning_rate"]}
- Scheduler: warmup={config["scheduler"]["warmup_steps"]}, min_lr_factor={config["scheduler"]["min_lr_factor"]}
- Batch size: {training_config["batch_size"]}
- Gradient accumulation steps: {training_config["gradient_accumulation_steps"]}
- Max grad norm: {training_config["max_grad_norm"]}

## Validation Curve

{curve}

## Sample Progression

{progression_block}

## Fixed Prompt Samples From Final Checkpoint

{sample_block}

## Resume Behavior

`checkpoint_last.pt` stores model state, optimizer state, current step, initial validation loss, best validation
loss, best step, and the full pretraining config. `python -m train.pretrain --resume` reloads that state from the
configured run directory and continues to the configured target step.

## Failure Modes

- The corpus is intentionally tiny and repo-authored, so sample quality is evidence of wiring and memorization
  behavior, not general language ability.
- Validation examples share the corpus style but are exact-hash separated from training records; remaining risk is
  distribution similarity, not duplicate leakage.
- Generated text may repeat because PHASE-04A does not add repetition penalties, nucleus sampling, or instruction
  tuning.

## Next Scale Recommendation

{eval_config["next_scale_recommendation"]}
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _sample_progression_block(sample_history: list[dict[str, Any]]) -> str:
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for row in sample_history:
        prompt = str(row.get("prompt", ""))
        by_prompt.setdefault(prompt, []).append(row)
    blocks: list[str] = []
    for prompt, rows in by_prompt.items():
        ordered = sorted(rows, key=lambda row: int(row.get("step", 0)))
        first = ordered[0]
        last = ordered[-1]
        blocks.append(
            f"### Prompt: `{prompt}`\n\n"
            f"- step {first['step']}: `{_short_text(_clean_text(str(first['generated_text'])))}`\n"
            f"- step {last['step']}: `{_short_text(_clean_text(str(last['generated_text'])))}`"
        )
    return "\n\n".join(blocks)


def _short_text(text: str, limit: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _clean_text(text: str) -> str:
    return "".join(char if char.isprintable() or char in {"\n", "\t"} else "�" for char in text)


if __name__ == "__main__":
    raise SystemExit(main())
