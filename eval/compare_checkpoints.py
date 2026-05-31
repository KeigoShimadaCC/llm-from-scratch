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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_comparison_report(
            checkpoint_manifest=checkpoint_manifest,
            eval_config=eval_config,
            results=results,
        ),
        encoding="utf8",
    )
    return {
        "output": str(output_path),
        "manifest": str(manifest_path),
        "eval_config": str(eval_config_path),
        "checkpoint_count": len(results),
        "live_evaluated_count": sum(1 for result in results if result["status"] == "live_evaluated"),
        "summary_only_count": sum(1 for result in results if result["status"] != "live_evaluated"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
