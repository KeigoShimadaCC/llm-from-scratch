from __future__ import annotations

import argparse
import json

from corpus.source_registry import SourceAuditError
from corpus.splitting import SplitManifestError, write_split_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create corpus_v01 split, dedup, and leakage manifest.")
    parser.add_argument("--config", required=True, help="Path to the corpus source registry YAML.")
    parser.add_argument("--processed", required=True, help="Processed corpus directory or documents.jsonl path.")
    parser.add_argument("--output", required=True, help="Path to write committed dataset manifest JSON.")
    parser.add_argument("--split-seed", type=int, default=1729, help="Deterministic split seed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = write_split_manifest(
            config_path=args.config,
            processed_path=args.processed,
            output_path=args.output,
            split_seed=args.split_seed,
        )
    except (SourceAuditError, SplitManifestError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
