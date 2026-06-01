from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kgpt.checkpoint import load_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare PHASE-05A scaling runs from a manifest.")
    parser.add_argument("--manifest", required=True, help="Scaling manifest JSON path.")
    parser.add_argument("--output", required=True, help="Markdown report output path.")
    args = parser.parse_args(argv)
    result = compare_runs(manifest_path=Path(args.manifest), output_path=Path(args.output))
    print(json.dumps(result, sort_keys=True))
    return 0


def compare_runs(*, manifest_path: Path, output_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    runs = [_resolve_run_summary(run) for run in manifest["runs"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_report(manifest=manifest, runs=runs), encoding="utf8")
    return {
        "output": str(output_path),
        "manifest": str(manifest_path),
        "run_count": len(runs),
        "trained_30m_present": any(
            run["parameter_count"] >= 30_000_000 and run["status"] == "trained" for run in runs
        ),
    }


def _resolve_run_summary(run: dict[str, Any]) -> dict[str, Any]:
    summary = dict(run["summary"])
    artifact_manifest = run.get("artifact_manifest")
    if isinstance(artifact_manifest, str) and Path(artifact_manifest).is_file():
        payload = json.loads(Path(artifact_manifest).read_text(encoding="utf8"))
        output_files = payload.get("output_files", {})
        summary.update(
            {
                "parameter_count": payload.get("parameter_count", summary.get("parameter_count")),
                "initial_validation_loss": payload.get(
                    "initial_validation_loss", summary.get("initial_validation_loss")
                ),
                "final_validation_loss": payload.get("final_validation_loss", summary.get("final_validation_loss")),
                "best_validation_loss": payload.get("best_validation_loss", summary.get("best_validation_loss")),
                "best_step": payload.get("best_step", summary.get("best_step")),
                "tokens_seen": payload.get("config", {}).get("run_budget", {}).get(
                    "target_tokens", summary.get("tokens_seen")
                ),
                "loss_improvement_fraction": payload.get(
                    "loss_improvement_fraction", summary.get("loss_improvement_fraction")
                ),
                "run_manifest_present": True,
                "run_output_files": output_files,
                "checkpoint_metadata": _checkpoint_metadata_summary(output_files.get("checkpoint_last")),
                "sample_snapshots": _sample_snapshots(output_files.get("samples")),
            }
        )
    else:
        summary.update(
            {
                "run_manifest_present": False,
                "run_output_files": {},
                "checkpoint_metadata": None,
                "sample_snapshots": [],
            }
        )
    return {**run, **summary}


def _render_report(*, manifest: dict[str, Any], runs: list[dict[str, Any]]) -> str:
    rows = "\n".join(
        "| {name} | {status} | {params:,} | {initial} | {final} | {improvement} | {tokens} |".format(
            name=run["name"],
            status=run["status"],
            params=int(run["parameter_count"]),
            initial=_fmt(run.get("initial_validation_loss")),
            final=_fmt(run.get("final_validation_loss")),
            improvement=_fmt_percent(run.get("loss_improvement_fraction")),
            tokens=run.get("tokens_seen", "n/a"),
        )
        for run in runs
    )
    stretch = "\n".join(f"- {item}" for item in manifest["stretch_decisions"])
    bottlenecks = "\n".join(f"- {item}" for item in manifest["bottlenecks"])
    evidence = _render_30m_evidence(runs)
    return f"""# PHASE-05A Scaling Report

## Summary

- 30M gate status: {manifest["gate_status"]}
- Final tokenizer: {manifest["tokenizer_decision"]["tokenizer_id"]}
- Data mixture: {manifest["data_mixture"]["ratio_summary"]}
- Profiling method: {manifest["profiling"]["method"]}

## Run Comparison

| Run | Status | Parameters | Initial loss | Final loss | Improvement | Tokens |
|---|---:|---:|---:|---:|---:|---:|
{rows}

## 30M Checkpoint Evidence

{evidence}

## Tokenizer Decision

{manifest["tokenizer_decision"]["rationale"]}

## Data Mixture

- Source manifest: `{manifest["data_mixture"]["manifest_path"]}`
- English: {manifest["data_mixture"]["english_ratio"]}
- Japanese: {manifest["data_mixture"]["japanese_ratio"]}
- Mixed English/Japanese: {manifest["data_mixture"]["mixed_ratio"]}
- License: {manifest["data_mixture"]["license"]}

## Mac Profiling

- Device: {manifest["profiling"]["device"]}
- Dtype: {manifest["profiling"]["dtype"]}
- Context length: {manifest["profiling"]["context_length"]}
- Batch size: {manifest["profiling"]["batch_size"]}
- Tokens/sec: {manifest["profiling"]["tokens_per_sec"]}
- Peak memory: {manifest["profiling"]["peak_memory"]}

## Bottlenecks

{bottlenecks}

## Stretch Decisions

{stretch}

## Phase Gate

{manifest["phase_gate"]}
"""


def _checkpoint_metadata_summary(checkpoint_path: Any) -> dict[str, Any] | None:
    if not isinstance(checkpoint_path, str) or not Path(checkpoint_path).is_file():
        return None
    metadata = load_checkpoint(checkpoint_path, map_location="cpu")["metadata"]
    return {
        "checkpoint": checkpoint_path,
        "model_name": metadata.get("model_name"),
        "step": metadata.get("step"),
        "tokens_seen": metadata.get("tokens_seen"),
        "parameter_count": metadata.get("parameter_count"),
        "current_validation_loss": metadata.get("current_validation_loss"),
        "best_validation_loss": metadata.get("best_validation_loss"),
        "best_step": metadata.get("best_step"),
        "config_hash": metadata.get("config_hash"),
        "git_commit": metadata.get("git_commit"),
    }


def _sample_snapshots(samples_path: Any) -> list[dict[str, Any]]:
    if not isinstance(samples_path, str) or not Path(samples_path).is_file():
        return []
    rows = [
        json.loads(line)
        for line in Path(samples_path).read_text(encoding="utf8").splitlines()
        if line.strip()
    ]
    if not rows:
        return []
    final_step = max(int(row.get("step", -1)) for row in rows)
    return [row for row in rows if int(row.get("step", -1)) == final_step]


def _render_30m_evidence(runs: list[dict[str, Any]]) -> str:
    run = next((item for item in runs if _numeric(item.get("parameter_count")) >= 30_000_000), None)
    if run is None:
        return "No 30M+ run is listed in the scaling manifest."
    metadata = run.get("checkpoint_metadata")
    samples = run.get("sample_snapshots") or []
    if metadata is None:
        reproduce_command = run.get(
            "reproduce_command",
            "uv run python -m train.pretrain --config configs/kgpt_30m.yaml --max-steps 40 "
            "--run-name phase05a_kgpt30m_smoke",
        )
        return (
            f"- Run: `{run['name']}`\n"
            "- Ignored checkpoint metadata was not loaded because local run artifacts are absent.\n"
            f"- Recreate with: `{reproduce_command}`"
        )
    sample_lines = "\n".join(
        "- step {step}, prompt `{prompt}`: `{generated}`".format(
            step=sample.get("step", "n/a"),
            prompt=sample.get("prompt", ""),
            generated=_clip(sample.get("generated_text", "")),
        )
        for sample in samples
    )
    if not sample_lines:
        sample_lines = "- No fixed-prompt sample snapshots were present in the local run directory."
    return f"""- Run: `{run["name"]}`
- Checkpoint: `{metadata["checkpoint"]}`
- Model name: `{metadata["model_name"]}`
- Parameters: {_fmt_int(metadata["parameter_count"])}
- Step: {metadata["step"]}
- Tokens seen: {metadata["tokens_seen"]}
- Current validation loss: {_fmt(metadata["current_validation_loss"])}
- Best validation loss: {_fmt(metadata["best_validation_loss"])} at step {metadata["best_step"]}
- Config hash: `{metadata["config_hash"]}`
- Git commit recorded in checkpoint: `{metadata["git_commit"]}`

Fixed-prompt sample snapshots:
{sample_lines}"""


def _fmt(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.4f}"
    return "n/a"


def _fmt_percent(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.2%}"
    return "n/a"


def _fmt_int(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{int(value):,}"
    return "n/a"


def _clip(value: Any, *, max_chars: int = 180) -> str:
    text = str(value).replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _numeric(value: Any) -> int:
    if isinstance(value, int | float):
        return int(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
