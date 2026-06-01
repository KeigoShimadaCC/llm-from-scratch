from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_COMMANDS = (
    "uv sync",
    "uv run pytest",
    "uv run ruff check .",
    "git diff --check",
    (
        "uv run python -m train.micro_char --config configs/micro_char.yaml --max-steps 200 "
        "--run-name phase01a_overfit_smoke"
    ),
    (
        "uv run python -m inference.generate_char --checkpoint "
        "experiments/runs/phase01a_overfit_smoke/checkpoint_last.pt --prompt hello --seed 123 --max-new-tokens 32"
    ),
    (
        "uv run python -m tokenizer.train_report --config configs/tokenizer_bilingual.yaml "
        "--output docs/tokenizer_report.md"
    ),
    "uv run python -m train.sample_batches --config configs/tokenized_smoke.yaml --max-batches 2",
    "uv run python -m train.transformer_smoke --config configs/transformer_micro.yaml --max-steps 20",
    (
        "uv run python -m inference.generate --config configs/transformer_micro.yaml --prompt hello "
        "--max-new-tokens 16 --seed 123"
    ),
    "uv run python -m train.pretrain --config configs/kgpt_tiny.yaml --max-steps 200 --run-name phase04a_tiny_smoke",
    (
        "uv run python -m eval.report --config configs/eval_fixed_prompts.yaml --checkpoint "
        "experiments/runs/phase04a_tiny_smoke/checkpoint_last.pt --output docs/phase04a_tiny_report.md"
    ),
    "uv run python -m train.pretrain --config configs/kgpt_30m.yaml --max-steps 40 --run-name phase05a_kgpt30m_smoke",
    "uv run python -m train.pretrain --config configs/kgpt_50m.yaml --dry-run",
    "uv run python -m train.pretrain --config configs/kgpt_100m.yaml --dry-run",
    (
        "uv run python -m eval.compare_runs --manifest docs/phase05a_scaling_manifest.json "
        "--output docs/phase05a_scaling_report.md"
    ),
    "uv run python -m train.sft --config configs/sft_smoke.yaml --max-steps 50 --run-name phase06a_sft_smoke",
    "uv run python -m eval.sft_compare --config configs/sft_eval.yaml --output docs/phase06a_sft_eval.md",
    "uv run python -m eval.report --config configs/eval_fixed_prompts.yaml --output docs/phase07a_eval_report.md",
    (
        "uv run python -m eval.compare_checkpoints --manifest docs/checkpoint_manifest.json "
        "--output docs/phase07a_checkpoint_comparison.md"
    ),
    (
        "uv run python -m inference.generate --config configs/inference_smoke.yaml --prompt hello "
        "--max-new-tokens 16 --seed 123"
    ),
    (
        "uv run python -m inference.chat --config configs/inference_smoke.yaml --instruction \"say hi\" "
        "--max-new-tokens 16 --seed 123"
    ),
    "uv run python -m inference.kv_cache_parity --config configs/inference_smoke.yaml",
    (
        "uv run python -m inference.benchmark --config configs/inference_benchmark.yaml --max-new-tokens 32 "
        "--output docs/phase08a_benchmark.md"
    ),
    "uv run python -m eval.audit_claims --doc docs/FINAL_WRITEUP.md --output docs/claim_evidence_audit.md",
    "uv run python -m eval.check_repro_commands --doc docs/COMMAND_INDEX.md",
)

FORBIDDEN_TOKENS = (
    "--max-new-chars",
    "phase01a_micro_char_overfit",
    "phase01a_char_overfit",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check that the command index covers required repro flows.")
    parser.add_argument("--doc", required=True, help="Command index markdown path.")
    args = parser.parse_args(argv)
    result = check_repro_commands(doc_path=Path(args.doc))
    print(json.dumps(result, sort_keys=True))
    if result["missing"] or result["forbidden_present"]:
        raise SystemExit(1)
    return 0


def check_repro_commands(*, doc_path: Path) -> dict[str, object]:
    text = doc_path.read_text(encoding="utf8")
    commands = _extract_bash_commands(text)
    command_set = set(commands)
    missing = [command for command in REQUIRED_COMMANDS if command not in command_set]
    forbidden_present = [token for token in FORBIDDEN_TOKENS if token in text]
    return {
        "doc": str(doc_path),
        "required_count": len(REQUIRED_COMMANDS),
        "command_count": len(commands),
        "missing": missing,
        "forbidden_present": forbidden_present,
    }


def _extract_bash_commands(text: str) -> list[str]:
    commands: list[str] = []
    in_bash = False
    pending = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            fence = line.removeprefix("```").strip()
            if in_bash:
                if pending:
                    commands.append(_normalize_shell_command(pending))
                    pending = ""
                in_bash = False
            else:
                in_bash = fence in {"bash", "sh", "shell"}
            continue
        if not in_bash or not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            pending += line[:-1].strip() + " "
            continue
        commands.append(_normalize_shell_command(pending + line))
        pending = ""
    return commands


def _normalize_shell_command(command: str) -> str:
    return " ".join(command.split())


if __name__ == "__main__":
    raise SystemExit(main())
