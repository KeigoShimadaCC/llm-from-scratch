from __future__ import annotations

import json
from pathlib import Path

from eval.checkpoint_eval import DEFAULT_SUMMARY_STATUS, evaluate_checkpoint_manifest, load_eval_config, repetition_rate
from eval.compare_checkpoints import compare_checkpoints


def test_eval_config_consolidates_prompt_sets() -> None:
    config = load_eval_config("configs/eval_fixed_prompts.yaml")

    categories = {prompt["category"] for prompt in config["fixed_prompts"]}
    assert {"english", "japanese", "technical", "instruction", "bilingual", "copy"} <= categories
    assert {task["expected_response"] for task in config["toy_tasks"]} == {"hi", "cat", "blue", "4"}
    assert {task["category"] for task in config["toy_tasks"]} >= {"toy_instruction", "arithmetic_toy"}


def test_summary_fallback_marks_missing_ignored_checkpoint() -> None:
    config = load_eval_config("configs/eval_fixed_prompts.yaml")
    manifest = {
        "schema_version": 1,
        "checkpoints": [
            {
                "id": "missing",
                "label": "Missing checkpoint",
                "phase": "TEST",
                "kind": "pretrain",
                "config": "configs/kgpt_tiny.yaml",
                "checkpoint": "experiments/runs/missing/checkpoint_last.pt",
                "summary": {
                    "parameter_count": 10,
                    "final_validation_loss": 2.0,
                    "exact_match_rate": 0.0,
                },
            }
        ],
    }

    result = evaluate_checkpoint_manifest(eval_config=config, checkpoint_manifest=manifest)[0]

    assert result["status"] == DEFAULT_SUMMARY_STATUS
    assert result["metrics"]["perplexity"] > 1.0
    assert result["failure_summary"] == ["live_samples_missing"]


def test_repetition_rate_detects_repeated_text() -> None:
    assert repetition_rate("ha ha ha ha ha") > repetition_rate("one two three four five")


def test_compare_checkpoints_writes_summary_report(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "comparison.md"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoints": [
                    {
                        "id": "missing",
                        "label": "Missing checkpoint",
                        "phase": "TEST",
                        "kind": "pretrain",
                        "config": "configs/kgpt_tiny.yaml",
                        "checkpoint": "experiments/runs/missing/checkpoint_last.pt",
                        "summary": {
                            "parameter_count": 10,
                            "final_validation_loss": 2.0,
                            "failure_summary": ["blocked_live_checkpoint_missing"],
                        },
                    }
                ],
            }
        ),
        encoding="utf8",
    )

    result = compare_checkpoints(
        manifest_path=manifest_path,
        eval_config_path=Path("configs/eval_fixed_prompts.yaml"),
        output_path=output_path,
    )

    assert result["summary_only_count"] == 1
    assert "Missing checkpoint" in output_path.read_text(encoding="utf8")
