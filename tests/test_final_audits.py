from __future__ import annotations

from pathlib import Path

from eval.audit_claims import audit_claims
from eval.check_repro_commands import REQUIRED_COMMANDS, check_repro_commands
from eval.report import _select_checkpoint_entry, _write_run_eval_report
from eval.sft_compare import _write_sft_run_eval_report


def test_repro_command_checker_requires_exact_commands(tmp_path: Path) -> None:
    command_doc = tmp_path / "COMMAND_INDEX.md"
    command_doc.write_text(
        "```bash\n"
        + "\n".join(command for command in REQUIRED_COMMANDS if "generate_char" not in command)
        + "\nuv run python -m inference.generate_char --config configs/micro_char.yaml "
        "--prompt hello --max-new-chars 16 --seed 123\n"
        + "```\n",
        encoding="utf8",
    )

    result = check_repro_commands(doc_path=command_doc)

    assert any("generate_char --checkpoint" in command for command in result["missing"])
    assert "--max-new-chars" in result["forbidden_present"]


def test_claim_audit_does_not_auto_support_ignored_run_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    writeup = docs / "FINAL_WRITEUP.md"
    output = docs / "audit.md"
    writeup.write_text(
        "| Claim | Evidence | Status |\n"
        "|---|---|---|\n"
        "| Unsupported ignored artifact | `experiments/runs/missing/checkpoint_last.pt` | Supported |\n",
        encoding="utf8",
    )

    result = audit_claims(doc_path=writeup, output_path=output)

    assert result["unsupported_count"] == 1
    assert "experiments/runs/missing/checkpoint_last.pt" in output.read_text(encoding="utf8")


def test_claim_audit_accepts_indexed_ignored_run_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ARTIFACT_INDEX.md").write_text(
        "| Path | Contents | Recreate with |\n"
        "|---|---|---|\n"
        "| `experiments/runs/indexed/` | Run dir. | "
        "`uv run python -m train.pretrain --config configs/kgpt_tiny.yaml` |\n",
        encoding="utf8",
    )
    (docs / "COMMAND_INDEX.md").write_text("", encoding="utf8")
    writeup = docs / "FINAL_WRITEUP.md"
    output = docs / "audit.md"
    writeup.write_text(
        "| Claim | Evidence | Status |\n"
        "|---|---|---|\n"
        "| Indexed ignored artifact | `experiments/runs/indexed/checkpoint_last.pt` | Supported |\n",
        encoding="utf8",
    )

    result = audit_claims(doc_path=writeup, output_path=output)

    assert result["unsupported_count"] == 0


def test_eval_report_records_run_manifest_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "phase04a_tiny_smoke"
    run_dir.mkdir()
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text('{"output_files": {"checkpoint_last": "checkpoint_last.pt"}}\n', encoding="utf8")

    report_path = _write_run_eval_report(checkpoint_path=run_dir / "checkpoint_last.pt", report_text="# Eval\n")

    assert report_path == run_dir / "eval_report.md"
    assert "eval_report" in manifest_path.read_text(encoding="utf8")


def test_single_checkpoint_entry_selection_prefers_exact_path() -> None:
    entries = [
        {"checkpoint": "experiments/runs/a/checkpoint_last.pt", "id": "a"},
        {"checkpoint": "experiments/runs/b/checkpoint_last.pt", "id": "b"},
    ]

    selected = _select_checkpoint_entry(
        checkpoint_entries=entries,
        checkpoint_path=Path("experiments/runs/b/checkpoint_last.pt"),
        manifest_path="docs/checkpoint_manifest.json",
    )

    assert selected == entries[1]


def test_sft_compare_records_run_manifest_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "phase06a_sft_smoke"
    run_dir.mkdir()
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text('{"output_files": {"checkpoint_last": "checkpoint_last.pt"}}\n', encoding="utf8")

    report_path = _write_sft_run_eval_report(checkpoint_path=run_dir / "checkpoint_last.pt", report_text="# Eval\n")

    assert report_path == run_dir / "eval_report.md"
    assert "eval_report" in manifest_path.read_text(encoding="utf8")
