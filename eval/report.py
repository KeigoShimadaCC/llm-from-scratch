from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.checkpoint_eval import (
    evaluate_checkpoint_entry,
    evaluate_checkpoint_manifest,
    load_checkpoint_manifest,
    load_eval_config,
    render_eval_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a comparable checkpoint evaluation report.")
    parser.add_argument("--config", required=True, help="Path to fixed-prompt eval YAML config.")
    parser.add_argument("--checkpoint", help="Optional single checkpoint path for backwards-compatible live eval.")
    parser.add_argument("--output", required=True, help="Markdown report output path.")
    args = parser.parse_args(argv)

    if args.checkpoint:
        result = generate_single_checkpoint_report(
            eval_config_path=Path(args.config),
            checkpoint_path=Path(args.checkpoint),
            output_path=Path(args.output),
        )
    else:
        result = generate_manifest_report(eval_config_path=Path(args.config), output_path=Path(args.output))
    print(json.dumps(result, sort_keys=True))
    return 0


def generate_manifest_report(*, eval_config_path: Path, output_path: Path) -> dict[str, Any]:
    eval_config = load_eval_config(eval_config_path)
    checkpoint_manifest = load_checkpoint_manifest(eval_config["checkpoint_manifest"])
    results = evaluate_checkpoint_manifest(eval_config=eval_config, checkpoint_manifest=checkpoint_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_eval_report(
            title=eval_config["title"],
            eval_config=eval_config,
            checkpoint_manifest=checkpoint_manifest,
            results=results,
        ),
        encoding="utf8",
    )
    return {
        "output": str(output_path),
        "eval_config": str(eval_config_path),
        "checkpoint_manifest": eval_config["checkpoint_manifest"],
        "checkpoint_count": len(results),
        "live_evaluated_count": sum(1 for result in results if result["status"] == "live_evaluated"),
        "summary_only_count": sum(1 for result in results if result["status"] != "live_evaluated"),
    }


def generate_single_checkpoint_report(
    *,
    eval_config_path: Path,
    checkpoint_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    eval_config = load_eval_config(eval_config_path)
    checkpoint_manifest = load_checkpoint_manifest(eval_config["checkpoint_manifest"])
    matching_entry = _select_checkpoint_entry(
        checkpoint_entries=checkpoint_manifest["checkpoints"],
        checkpoint_path=checkpoint_path,
        manifest_path=eval_config["checkpoint_manifest"],
    )
    if matching_entry is None:
        raise ValueError(
            "Single-checkpoint eval requires the checkpoint to be listed in "
            f"{eval_config['checkpoint_manifest']}: {checkpoint_path}"
        )
    result = evaluate_checkpoint_entry(
        entry={**matching_entry, "checkpoint": str(checkpoint_path)},
        eval_config=eval_config,
    )
    eval_report_text = render_eval_report(
        title=f"{eval_config['title']} - {result['label']}",
        eval_config=eval_config,
        checkpoint_manifest=checkpoint_manifest,
        results=[result],
    )
    report_text = _render_single_checkpoint_output(
        entry=matching_entry,
        checkpoint_path=checkpoint_path,
        result=result,
        eval_report_text=eval_report_text,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf8")
    run_eval_report = _write_run_eval_report(checkpoint_path=checkpoint_path, report_text=eval_report_text)
    return {
        "output": str(output_path),
        "checkpoint": str(checkpoint_path),
        "status": result["status"],
        "parameter_count": result.get("parameter_count"),
        "validation_loss": result["metrics"].get("validation_loss"),
        "run_eval_report": str(run_eval_report) if run_eval_report else None,
    }


def _select_checkpoint_entry(
    *,
    checkpoint_entries: list[dict[str, Any]],
    checkpoint_path: Path,
    manifest_path: str,
) -> dict[str, Any] | None:
    exact_matches = [
        entry for entry in checkpoint_entries if Path(entry["checkpoint"]) == checkpoint_path
    ]
    if exact_matches:
        return exact_matches[0]
    basename_matches = [
        entry for entry in checkpoint_entries if Path(entry["checkpoint"]).name == checkpoint_path.name
    ]
    if len(basename_matches) > 1:
        raise ValueError(
            "Single-checkpoint eval basename is ambiguous in "
            f"{manifest_path}; pass a listed checkpoint path instead: {checkpoint_path}"
        )
    return basename_matches[0] if basename_matches else None


def _render_single_checkpoint_output(
    *,
    entry: dict[str, Any],
    checkpoint_path: Path,
    result: dict[str, Any],
    eval_report_text: str,
) -> str:
    if entry.get("id") == "phase04a_tiny":
        return _render_phase04_training_report(checkpoint_path=checkpoint_path, result=result)
    return eval_report_text


def _render_phase04_training_report(*, checkpoint_path: Path, result: dict[str, Any]) -> str:
    run_dir = checkpoint_path.parent
    manifest = _load_json(run_dir / "manifest.json")
    config = manifest["config"]
    data_manifest = _load_json(Path("docs/phase04a_data_manifest.json"))
    tokenizer_info = _load_json(run_dir / "tokenizer_info.json")
    metric_rows = _load_jsonl(Path(manifest["output_files"]["metrics"]))
    sample_rows = _load_jsonl(Path(manifest["output_files"]["samples"]))
    validation_rows = [row for row in metric_rows if row.get("validation_loss") is not None]
    validation_curve = "\n".join(
        "- step {step}: validation_loss={loss}, perplexity={perplexity}".format(
            step=row["step"],
            loss=_fmt(row["validation_loss"]),
            perplexity=_fmt(row.get("perplexity")),
        )
        for row in validation_rows
    )
    sample_progression = _render_sample_progression(sample_rows)
    failure_modes = ", ".join(result["failure_summary"]) or "none"
    model = config["model"]
    training = config["training"]
    run_budget = config["run_budget"]
    return f"""# PHASE-04A tiny pretraining fixed-prompt report

## Summary

- Checkpoint: `{checkpoint_path}`
- Model: `{config["model_name"]}`
- Parameter count: {manifest["parameter_count"]:,}
- Initial validation loss: {_fmt(manifest["initial_validation_loss"])}
- Final validation loss: {_fmt(manifest["final_validation_loss"])}
- Best validation loss: {_fmt(manifest["best_validation_loss"])} at step {manifest["best_step"]}
- Loss improvement: {_fmt_percent(manifest["loss_improvement_fraction"])}
- Predeclared threshold: {_fmt_percent(manifest["loss_improvement_threshold"])}
- Gate: {manifest["validation_status"]}

## Dataset Source And License

- Source: {data_manifest["source_name"]}
- License: {data_manifest["license"]}
- Language mix: {data_manifest["language_mix"]}
- Split method: {data_manifest["split_method"]}
- Dedup strategy: {data_manifest["dedup_strategy"]}
- Leakage check: {data_manifest["leakage_check"]["overlap_count"]} overlapping normalized text hashes.

## Tokenizer

- Tokenizer id: phase02a-byte-bpe-bilingual
- Algorithm: {tokenizer_info["algorithm"]}
- Vocabulary size: {model["vocab_size"]}
- Byte fallback: {tokenizer_info["byte_fallback"]}

## Model Config

- Context length: {model["context_length"]}
- Embedding dimension: {model["embedding_dim"]}
- Layers: {model["num_layers"]}
- Heads: {model["num_heads"]}
- MLP hidden dimension: {model["mlp_hidden_dim"]}
- Tied embeddings: {model["tie_embeddings"]}

## Run Budget And Hyperparameters

- Target steps: {run_budget["target_steps"]}
- Target tokens: {run_budget["target_tokens"]}
- Target wall clock: {run_budget["target_wall_clock_minutes"]} minutes
- Hardware/device: {run_budget["hardware"]} / `{manifest["device"]}`
- Optimizer: {config["optimizer"]["name"]}, lr={config["optimizer"]["learning_rate"]}
- Scheduler: warmup={config["scheduler"]["warmup_steps"]}, min_lr_factor={config["scheduler"]["min_lr_factor"]}
- Batch size: {training["batch_size"]}
- Gradient accumulation steps: {training["gradient_accumulation_steps"]}
- Max grad norm: {training["max_grad_norm"]}

## Validation Curve

{validation_curve}

## Sample Progression

{sample_progression}

## Fixed Prompt Eval Addendum

- Eval status: {result["status"]}
- Eval validation loss: {_fmt(result["metrics"].get("validation_loss"))}
- Eval perplexity: {_fmt(result["metrics"].get("perplexity"))}
- Toy exact match: {_fmt_percent(result["metrics"].get("exact_match_rate"))}
- Failure classes: {failure_modes}
- Run-local eval report: `experiments/runs/phase04a_tiny_smoke/eval_report.md`

## Resume Behavior

- `checkpoint_last.pt` and `checkpoint_best.pt` are both produced.
- The pretraining path supports resume through `checkpoint_last.pt`; the phase smoke run was generated from config.

## Failure Modes

- The fixture corpus is intentionally tiny and can encourage memorization.
- Japanese generation still shows byte-level token boundary artifacts.
- The model is useful as training-loop evidence, not as a general language model.

## Next Scale Recommendation

Proceed to PHASE-05A only as a scale-gate experiment with the same artifact policy and explicit quality limitations.
"""


def _write_run_eval_report(*, checkpoint_path: Path, report_text: str) -> Path | None:
    manifest_path = checkpoint_path.parent / "manifest.json"
    if not manifest_path.is_file():
        return None
    eval_report_path = checkpoint_path.parent / "eval_report.md"
    eval_report_path.write_text(report_text, encoding="utf8")
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    output_files = manifest.setdefault("output_files", {})
    output_files["eval_report"] = str(eval_report_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf8")
    return eval_report_path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf8").splitlines()
        if line.strip()
    ]


def _render_sample_progression(rows: list[dict[str, Any]]) -> str:
    prompts = []
    for row in rows:
        prompt = row["prompt"]
        if prompt not in prompts:
            prompts.append(prompt)
    blocks = []
    for prompt in prompts:
        prompt_rows = [row for row in rows if row["prompt"] == prompt]
        first = prompt_rows[0]
        last = prompt_rows[-1]
        blocks.append(
            "### Prompt: `{prompt}`\n\n"
            "- step {first_step}: `{first_text}`\n"
            "- step {last_step}: `{last_text}`".format(
                prompt=prompt,
                first_step=first["step"],
                first_text=_clip(first["generated_text"]),
                last_step=last["step"],
                last_text=_clip(last["generated_text"]),
            )
        )
    return "\n\n".join(blocks)


def _fmt(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.4f}"
    return "n/a"


def _fmt_percent(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.2%}"
    return "n/a"


def _clip(value: Any, *, max_chars: int = 180) -> str:
    text = str(value).replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
