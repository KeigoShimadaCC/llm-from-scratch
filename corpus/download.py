from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from corpus.source_registry import SourceAuditError, audit_config, load_corpus_config


class DownloadPlanError(ValueError):
    """Raised when corpus download planning cannot proceed."""


def build_download_plan(config_path: str | Path) -> dict[str, Any]:
    audit = audit_config(config_path)
    config = load_corpus_config(config_path)
    sources = sorted(config["sources"], key=lambda source: source["id"])
    entries: list[dict[str, Any]] = []
    for source in sources:
        provenance = source["provenance"]
        checksum = source["checksum"]
        storage = source["storage"]
        locator = (
            provenance.get("download_url_template")
            or provenance.get("catalog_url")
            or provenance.get("item_landing_url_template")
        )
        entries.append(
            {
                "source_id": source["id"],
                "language": source["language"],
                "status": source["status"],
                "eligible_for_phase_10b": source["eligible_for_phase_10b"],
                "provider": provenance["provider"],
                "locator": locator,
                "terms_url": provenance["terms_url"],
                "license_url": provenance["license_url"],
                "raw_path": storage["raw_path"],
                "processed_path": storage["processed_path"],
                "checksum_required": checksum["required"],
                "checksum_algorithm": checksum["algorithm"],
                "checksum_policy": checksum["policy"],
                "action": "would_plan_download" if source["eligible_for_phase_10b"] else "blocked",
                "notes": provenance.get("notes", ""),
            }
        )

    return {
        "schema_version": 1,
        "mode": "dry_run",
        "config_path": str(config_path),
        "config_sha256": audit.config_hash,
        "corpus_id": config["corpus_id"],
        "payload_downloads_performed": False,
        "source_count": len(entries),
        "sources": entries,
        "blockers": [
            f"{entry['source_id']} is not eligible for PHASE-10B"
            for entry in entries
            if entry["action"] == "blocked"
        ],
    }


def run_download(config_path: str | Path, *, dry_run: bool) -> dict[str, Any]:
    plan = build_download_plan(config_path)
    if dry_run:
        return plan
    raise DownloadPlanError(
        "Full corpus payload downloads are not enabled by PHASE-10B automation. "
        "Run the dry-run plan first, then use a later supervised local download path with concrete "
        "dump dates, item selections, checksums, and storage limits."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan corpus_v01 downloads without committing payloads.")
    parser.add_argument("--config", required=True, help="Path to the corpus source registry YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the download plan only.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = run_download(args.config, dry_run=args.dry_run)
    except (DownloadPlanError, SourceAuditError) as exc:
        parser.error(str(exc))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
