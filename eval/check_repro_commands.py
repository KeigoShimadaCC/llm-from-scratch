from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_COMMAND_SNIPPETS = (
    "uv sync",
    "uv run pytest",
    "uv run ruff check .",
    "git diff --check",
    "uv run python -m train.micro_char",
    "uv run python -m tokenizer.train_report",
    "uv run python -m train.sample_batches",
    "uv run python -m train.transformer_smoke",
    "uv run python -m train.pretrain --config configs/kgpt_tiny.yaml",
    "uv run python -m train.pretrain --config configs/kgpt_30m.yaml",
    "uv run python -m train.sft",
    "uv run python -m eval.report",
    "uv run python -m eval.compare_checkpoints",
    "uv run python -m inference.generate",
    "uv run python -m inference.kv_cache_parity",
    "uv run python -m inference.benchmark",
    "uv run python -m eval.audit_claims",
    "uv run python -m eval.check_repro_commands",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check that the command index covers required repro flows.")
    parser.add_argument("--doc", required=True, help="Command index markdown path.")
    args = parser.parse_args(argv)
    result = check_repro_commands(doc_path=Path(args.doc))
    print(json.dumps(result, sort_keys=True))
    if result["missing"]:
        raise SystemExit(1)
    return 0


def check_repro_commands(*, doc_path: Path) -> dict[str, object]:
    text = doc_path.read_text(encoding="utf8")
    missing = [snippet for snippet in REQUIRED_COMMAND_SNIPPETS if snippet not in text]
    return {
        "doc": str(doc_path),
        "required_count": len(REQUIRED_COMMAND_SNIPPETS),
        "missing": missing,
    }


if __name__ == "__main__":
    raise SystemExit(main())
