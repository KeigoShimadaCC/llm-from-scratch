from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

DEFAULT_REPO_URL = "https://github.com/KeigoShimadaCC/llm-from-scratch.wiki.git"
DEFAULT_WORKDIR = Path("/private/tmp/llm-from-scratch-wiki")
DEFAULT_BRANCH = "master"
DEFAULT_MESSAGE = "Add LLM from scratch course wiki"
EXCLUDED_SOURCE_FILES = {"README.md"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    allow_failure: bool = False,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0 and not allow_failure:
        location = f" in {cwd}" if cwd else ""
        raise RuntimeError(
            f"Command failed{location}: {' '.join(args)}\n"
            f"stdout:\n{result.stdout.strip()}\n"
            f"stderr:\n{result.stderr.strip()}"
        )
    return result


def _git(workdir: Path, *args: str, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], cwd=workdir, allow_failure=allow_failure)


def _status(workdir: Path) -> str:
    return _git(workdir, "status", "--porcelain", "--untracked-files=all").stdout.strip()


def _status_paths(status: str) -> set[str]:
    paths: set[str] = set()
    for line in status.splitlines():
        if not line:
            continue
        path = line[2:].strip()
        if " -> " in path:
            path = path.rsplit(" -> ", maxsplit=1)[1]
        paths.add(path)
    return paths


def _tracked_files(workdir: Path) -> set[str]:
    output = _git(workdir, "ls-files").stdout
    return {line for line in output.splitlines() if line}


def _ensure_clean(workdir: Path) -> None:
    status = _status(workdir)
    if status:
        raise RuntimeError(
            f"Wiki worktree is dirty; commit, stash, or remove local changes before publishing.\n{status}"
        )


def _git_config(workdir: Path, key: str) -> str | None:
    result = _git(workdir, "config", "--get", key, allow_failure=True)
    value = result.stdout.strip()
    return value or None


def _set_default_identity(workdir: Path) -> None:
    if _git_config(workdir, "user.name") is None:
        _git(workdir, "config", "user.name", "Codex Wiki Publisher")
    if _git_config(workdir, "user.email") is None:
        _git(workdir, "config", "user.email", "codex-wiki-publisher@example.invalid")


def _clone_or_update(workdir: Path, repo_url: str, branch: str) -> None:
    if not workdir.exists():
        workdir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--branch", branch, repo_url, str(workdir)])
        return

    if not (workdir / ".git").exists():
        raise RuntimeError(f"Wiki workdir exists but is not a git repository: {workdir}")

    remote_url = _git(workdir, "config", "--get", "remote.origin.url").stdout.strip()
    if remote_url != repo_url:
        raise RuntimeError(
            f"Wiki workdir points at a different remote.\nexpected: {repo_url}\nactual: {remote_url}"
        )

    _ensure_clean(workdir)
    _git(workdir, "fetch", "origin", branch)
    _git(workdir, "checkout", branch)
    _git(workdir, "pull", "--ff-only", "origin", branch)
    _ensure_clean(workdir)


def managed_pages(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Wiki source directory does not exist: {source_dir}")
    pages = sorted(path for path in source_dir.glob("*.md") if path.name not in EXCLUDED_SOURCE_FILES)
    if not pages:
        raise RuntimeError(f"No managed wiki pages found in {source_dir}")
    return pages


def _copy_pages(pages: Sequence[Path], workdir: Path) -> list[str]:
    copied: list[str] = []
    for page in pages:
        destination = workdir / page.name
        shutil.copy2(page, destination)
        copied.append(page.name)
    return copied


def _reset_dry_run(workdir: Path, page_names: Sequence[str], tracked_before: set[str]) -> None:
    _git(workdir, "reset", "--hard", "HEAD")
    for page_name in page_names:
        destination = workdir / page_name
        if page_name not in tracked_before and destination.exists():
            destination.unlink()
    _ensure_clean(workdir)


def publish_wiki(
    *,
    source_dir: Path,
    workdir: Path,
    repo_url: str,
    branch: str,
    message: str,
    dry_run: bool,
    push: bool,
) -> dict[str, object]:
    if dry_run == push:
        raise ValueError("Exactly one of dry_run or push must be true.")

    pages = managed_pages(source_dir)
    page_names = [page.name for page in pages]
    managed_page_names = set(page_names)

    _clone_or_update(workdir=workdir, repo_url=repo_url, branch=branch)
    tracked_before = _tracked_files(workdir)
    _copy_pages(pages=pages, workdir=workdir)

    status = _status(workdir)
    changed_paths = _status_paths(status)
    unmanaged_changes = changed_paths - managed_page_names
    if unmanaged_changes:
        if dry_run:
            _reset_dry_run(workdir=workdir, page_names=page_names, tracked_before=tracked_before)
        raise RuntimeError(f"Refusing to publish because unmanaged files changed: {sorted(unmanaged_changes)}")

    result: dict[str, object] = {
        "branch": branch,
        "dry_run": dry_run,
        "pages": page_names,
        "status": status.splitlines(),
        "would_commit": bool(status),
    }

    if dry_run:
        _reset_dry_run(workdir=workdir, page_names=page_names, tracked_before=tracked_before)
        return result

    if not status:
        result["pushed"] = False
        result["reason"] = "no_changes"
        return result

    _set_default_identity(workdir)
    _git(workdir, "add", *page_names)
    _git(workdir, "commit", "-m", message)
    _git(workdir, "push", "origin", branch)
    result["pushed"] = True
    result["message"] = message
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish managed docs/wiki pages to the GitHub Wiki repository.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare pages and report changes without committing or pushing.",
    )
    mode.add_argument("--push", action="store_true", help="Commit and push managed wiki page changes.")
    parser.add_argument("--message", default=DEFAULT_MESSAGE, help="Commit message used with --push.")
    parser.add_argument("--source-dir", type=Path, default=_repo_root() / "docs" / "wiki")
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = publish_wiki(
        source_dir=args.source_dir,
        workdir=args.workdir,
        repo_url=args.repo_url,
        branch=args.branch,
        message=args.message,
        dry_run=args.dry_run,
        push=args.push,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
