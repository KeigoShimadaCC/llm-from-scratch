from __future__ import annotations

import argparse
import json
from pathlib import Path

from corpus.cleaning import write_smoke_corpus
from corpus.source_registry import SourceAuditError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean corpus_v01 text into processed JSONL.")
    parser.add_argument("--config", required=True, help="Path to the corpus source registry YAML.")
    parser.add_argument("--smoke", action="store_true", help="Generate the repo-authored smoke processed corpus.")
    parser.add_argument("--output", required=True, help="Ignored output directory for processed corpus artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.smoke:
        parser.error("PHASE-10B automation supports only --smoke cleaning; full cleaning is deferred.")
    try:
        manifest = write_smoke_corpus(args.config, Path(args.output))
    except (SourceAuditError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
