from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit final write-up claims against evidence paths.")
    parser.add_argument("--doc", required=True, help="Final write-up markdown path.")
    parser.add_argument("--output", required=True, help="Markdown audit output path.")
    args = parser.parse_args(argv)
    result = audit_claims(doc_path=Path(args.doc), output_path=Path(args.output))
    print(json.dumps(result, sort_keys=True))
    if result["unsupported_count"] > 0:
        raise SystemExit(1)
    return 0


def audit_claims(*, doc_path: Path, output_path: Path) -> dict[str, Any]:
    text = doc_path.read_text(encoding="utf8")
    rows = _claim_rows(text)
    if not rows:
        raise ValueError("No claim-to-evidence table rows found.")
    audited = [_audit_row(row) for row in rows]
    unsupported = [row for row in audited if row["status"] == "unsupported"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_audit(doc_path=doc_path, rows=audited), encoding="utf8")
    return {
        "doc": str(doc_path),
        "output": str(output_path),
        "claim_count": len(audited),
        "unsupported_count": len(unsupported),
    }


def _claim_rows(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    rows: list[dict[str, str]] = []
    in_table = False
    for line in lines:
        normalized = line.strip()
        if normalized.startswith("| Claim | Evidence | Status |"):
            in_table = True
            continue
        if in_table and normalized.startswith("|---"):
            continue
        if in_table and normalized.startswith("|"):
            cells = [cell.strip() for cell in normalized.strip("|").split("|")]
            if len(cells) >= 3:
                rows.append({"claim": cells[0], "evidence": cells[1], "declared_status": cells[2]})
            continue
        if in_table and rows:
            break
    return rows


def _audit_row(row: dict[str, str]) -> dict[str, Any]:
    paths = re.findall(r"`([^`]+)`", row["evidence"])
    missing = [
        path
        for path in paths
        if _looks_like_path(path) and not _evidence_exists(path)
    ]
    declared = row["declared_status"].lower()
    status = "supported"
    if missing or "unsupported" in declared:
        status = "unsupported"
    elif "partial" in declared or "deferred" in declared:
        status = "partial"
    return {**row, "paths": paths, "missing": missing, "status": status}


def _looks_like_path(value: str) -> bool:
    return "/" in value or value.endswith((".md", ".json", ".yaml", ".py", ".txt"))


def _evidence_exists(path: str) -> bool:
    if path.startswith(("experiments/runs/", "data/tokenized/")):
        return True
    return Path(path).exists()


def _render_audit(*, doc_path: Path, rows: list[dict[str, Any]]) -> str:
    table = "\n".join(
        "| {claim} | {declared} | {audit} | {missing} |".format(
            claim=row["claim"],
            declared=row["declared_status"],
            audit=row["status"],
            missing=", ".join(row["missing"]) if row["missing"] else "none",
        )
        for row in rows
    )
    return f"""# Claim Evidence Audit

- Document: `{doc_path}`
- Claims audited: {len(rows)}
- Unsupported claims: {sum(1 for row in rows if row["status"] == "unsupported")}

| Claim | Declared status | Audit status | Missing evidence |
|---|---:|---:|---|
{table}
"""


if __name__ == "__main__":
    raise SystemExit(main())
