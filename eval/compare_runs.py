from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def _fmt(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.4f}"
    return "n/a"


def _fmt_percent(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.2%}"
    return "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
