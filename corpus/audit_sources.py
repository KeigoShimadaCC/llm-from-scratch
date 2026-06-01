from __future__ import annotations

import argparse
from pathlib import Path

from corpus.source_registry import SourceAuditError, audit_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit corpus_v01 source registry metadata without downloads.")
    parser.add_argument("--config", required=True, help="Path to the corpus source registry YAML.")
    parser.add_argument("--output", required=True, help="Path for the generated Markdown source manifest.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = audit_config(args.config)
    except SourceAuditError as exc:
        parser.error(str(exc))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.manifest, encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
