from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.publish_wiki import managed_pages, publish_wiki

REPO_ROOT = Path(__file__).resolve().parents[1]
WIKI_SOURCE = REPO_ROOT / "docs" / "wiki"
REPO_LINK = "https://github.com/KeigoShimadaCC/llm-from-scratch"


def _git(workdir: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=workdir,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed in {workdir}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def _seed_wiki_remote(tmp_path: Path) -> Path:
    remote = tmp_path / "wiki.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)

    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", str(remote), str(seed)], check=True, capture_output=True)
    _git(seed, "checkout", "-b", "master")
    _git(seed, "config", "user.name", "Test Publisher")
    _git(seed, "config", "user.email", "test@example.invalid")
    (seed / "Home.md").write_text("Welcome to the llm-from-scratch wiki!\n", encoding="utf8")
    (seed / "Extra.md").write_text("# Existing Page\n\nPreserve this page.\n", encoding="utf8")
    _git(seed, "add", "Home.md", "Extra.md")
    _git(seed, "commit", "-m", "Seed wiki")
    _git(seed, "push", "origin", "master")
    return remote


def _write_source_pages(source_dir: Path) -> None:
    source_dir.mkdir()
    (source_dir / "Home.md").write_text("# Course Home\n", encoding="utf8")
    (source_dir / "_Sidebar.md").write_text("* [Home](Home)\n", encoding="utf8")
    (source_dir / "README.md").write_text("# Source policy\n", encoding="utf8")


def test_managed_pages_excludes_source_readme() -> None:
    page_names = {path.name for path in managed_pages(WIKI_SOURCE)}

    assert "README.md" not in page_names
    assert "Home.md" in page_names
    assert "_Sidebar.md" in page_names


def test_wiki_source_pages_have_course_structure() -> None:
    pages = managed_pages(WIKI_SOURCE)
    sidebar = (WIKI_SOURCE / "_Sidebar.md").read_text(encoding="utf8")
    home = (WIKI_SOURCE / "Home.md").read_text(encoding="utf8")

    for page in pages:
        if page.name != "_Sidebar.md":
            assert page.stem in sidebar

        if page.name not in {"_Sidebar.md", "Home.md"}:
            assert page.stem in home

        text = page.read_text(encoding="utf8")
        assert text.startswith("# ")

        if page.name != "_Sidebar.md":
            assert REPO_LINK in text

        if page.name not in {"_Sidebar.md", "Home.md", "Appendix-Command-Index.md"}:
            assert "## Goal" in text
            assert "## What This Part Does" in text
            assert "## Run It" in text
            assert "## Further Reading" in text


def test_wiki_gap_fixes_are_documented() -> None:
    tokenizer_page = (WIKI_SOURCE / "01-Text-To-Tokens.md").read_text(encoding="utf8")
    labs_page = (WIKI_SOURCE / "10-Hands-On-Labs.md").read_text(encoding="utf8")

    assert "English sentence" in tokenizer_page
    assert "Japanese sentence" in tokenizer_page
    assert "0 unknown tokens" in tokenizer_page
    assert "Lab 6: Inspect Checkpoint Comparison" in labs_page
    assert "docs/checkpoint_manifest_corpus_v01.json" in labs_page


def test_publish_wiki_dry_run_preserves_remote_and_cleans_workdir(tmp_path: Path) -> None:
    remote = _seed_wiki_remote(tmp_path)
    source_dir = tmp_path / "source"
    workdir = tmp_path / "workdir"
    _write_source_pages(source_dir)

    result = publish_wiki(
        source_dir=source_dir,
        workdir=workdir,
        repo_url=str(remote),
        branch="master",
        message="Publish wiki",
        dry_run=True,
        push=False,
    )

    assert result["dry_run"] is True
    assert result["would_commit"] is True
    assert "Home.md" in result["pages"]
    assert _git(workdir, "status", "--porcelain", "--untracked-files=all") == ""

    verify = tmp_path / "verify-dry-run"
    subprocess.run(["git", "clone", str(remote), str(verify)], check=True, capture_output=True)
    assert (verify / "Home.md").read_text(encoding="utf8") == "Welcome to the llm-from-scratch wiki!\n"
    assert not (verify / "_Sidebar.md").exists()


def test_publish_wiki_push_updates_managed_pages_and_preserves_unmanaged(tmp_path: Path) -> None:
    remote = _seed_wiki_remote(tmp_path)
    source_dir = tmp_path / "source"
    workdir = tmp_path / "workdir"
    _write_source_pages(source_dir)

    result = publish_wiki(
        source_dir=source_dir,
        workdir=workdir,
        repo_url=str(remote),
        branch="master",
        message="Publish wiki",
        dry_run=False,
        push=True,
    )

    assert result["pushed"] is True

    verify = tmp_path / "verify-push"
    subprocess.run(["git", "clone", str(remote), str(verify)], check=True, capture_output=True)
    assert (verify / "Home.md").read_text(encoding="utf8") == "# Course Home\n"
    assert (verify / "_Sidebar.md").read_text(encoding="utf8") == "* [Home](Home)\n"
    assert (verify / "Extra.md").read_text(encoding="utf8") == "# Existing Page\n\nPreserve this page.\n"
    assert not (verify / "README.md").exists()


def test_publish_wiki_refuses_dirty_existing_workdir(tmp_path: Path) -> None:
    remote = _seed_wiki_remote(tmp_path)
    source_dir = tmp_path / "source"
    workdir = tmp_path / "workdir"
    _write_source_pages(source_dir)

    publish_wiki(
        source_dir=source_dir,
        workdir=workdir,
        repo_url=str(remote),
        branch="master",
        message="Publish wiki",
        dry_run=True,
        push=False,
    )

    (workdir / "Extra.md").write_text("# Existing Page\n\nDirty local edit.\n", encoding="utf8")

    with pytest.raises(RuntimeError, match="dirty"):
        publish_wiki(
            source_dir=source_dir,
            workdir=workdir,
            repo_url=str(remote),
            branch="master",
            message="Publish wiki",
            dry_run=True,
            push=False,
        )
