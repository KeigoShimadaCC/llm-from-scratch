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
    matching_entries = [
        entry
        for entry in checkpoint_manifest["checkpoints"]
        if Path(entry["checkpoint"]) == checkpoint_path or Path(entry["checkpoint"]).name == checkpoint_path.name
    ]
    if not matching_entries:
        raise ValueError(
            "Single-checkpoint eval requires the checkpoint to be listed in "
            f"{eval_config['checkpoint_manifest']}: {checkpoint_path}"
        )
    result = evaluate_checkpoint_entry(
        entry={**matching_entries[0], "checkpoint": str(checkpoint_path)},
        eval_config=eval_config,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_eval_report(
            title=f"{eval_config['title']} - {result['label']}",
            eval_config=eval_config,
            checkpoint_manifest=checkpoint_manifest,
            results=[result],
        ),
        encoding="utf8",
    )
    return {
        "output": str(output_path),
        "checkpoint": str(checkpoint_path),
        "status": result["status"],
        "parameter_count": result.get("parameter_count"),
        "validation_loss": result["metrics"].get("validation_loss"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
