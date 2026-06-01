from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.checkpoint_eval import (
    evaluate_checkpoint_manifest,
    load_checkpoint_manifest,
    load_eval_config,
    render_comparison_report,
)
from eval.report import _write_run_eval_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare compatible checkpoints with the PHASE-07A eval schema.")
    parser.add_argument("--manifest", required=True, help="Checkpoint manifest JSON path.")
    parser.add_argument("--output", required=True, help="Markdown report output path.")
    parser.add_argument(
        "--eval-config",
        default="configs/eval_fixed_prompts.yaml",
        help="Fixed-prompt eval YAML config.",
    )
    args = parser.parse_args(argv)
    result = compare_checkpoints(
        manifest_path=Path(args.manifest),
        eval_config_path=Path(args.eval_config),
        output_path=Path(args.output),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def compare_checkpoints(
    *,
    manifest_path: Path,
    eval_config_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    eval_config = load_eval_config(eval_config_path)
    checkpoint_manifest = load_checkpoint_manifest(manifest_path)
    results = evaluate_checkpoint_manifest(eval_config=eval_config, checkpoint_manifest=checkpoint_manifest)
    report_text = render_comparison_report(
        checkpoint_manifest=checkpoint_manifest,
        eval_config=eval_config,
        results=results,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf8")
    run_eval_reports = _write_live_run_eval_reports(results=results, report_text=report_text)
    return {
        "output": str(output_path),
        "manifest": str(manifest_path),
        "eval_config": str(eval_config_path),
        "checkpoint_count": len(results),
        "live_evaluated_count": sum(1 for result in results if result["status"] == "live_evaluated"),
        "summary_only_count": sum(1 for result in results if result["status"] != "live_evaluated"),
        "run_eval_reports": run_eval_reports,
    }


def _write_live_run_eval_reports(*, results: list[dict[str, Any]], report_text: str) -> list[str]:
    report_paths = []
    for result in results:
        if result["status"] != "live_evaluated":
            continue
        report_path = _write_run_eval_report(checkpoint_path=Path(result["checkpoint"]), report_text=report_text)
        if report_path is not None:
            report_paths.append(str(report_path))
    return report_paths


if __name__ == "__main__":
    raise SystemExit(main())
