# PHASE-00B - Repository And Lab Foundation

## Goal

Implement Phase 0 from `docs/NORTH_STAR_LLM_FROM_SCRATCH_MAC.md`: create the clean Python research-engineering foundation for a reproducible Mac-local LLM lab.

## Scope

- Create the initial repo skeleton for configs, data docs, experiments docs, model package, training, evaluation, inference, tokenizer, and tests.
- Add `pyproject.toml` and `uv.lock` for a Python 3.11/3.12 project using PyTorch-first tooling.
- Add deterministic seeding, config loading, dummy fake-token data, checkpoint save/load, and experiment logging conventions.
- Add a tiny dummy training command that runs without private data and writes artifacts only under ignored run directories.
- Add minimal unit tests for config loading, seeding, checkpoint roundtrip, and dummy training.
- Update root docs and `PROGRESS.md`.

## Prerequisites

- `docs/NORTH_STAR_LLM_FROM_SCRATCH_MAC.md` has been read and remains the source of truth.
- The agentic runner is available through `./bin/agentic`.
- No Python implementation is assumed to exist before this phase.

## Phase Dependencies

- Depends on PHASE-00A standards and the North Star doc.
- Unblocks PHASE-01A by creating the repo skeleton, config loader, checkpoint helpers, and artifact conventions.

## Allowed Paths

- `.gitignore`
- `README.md`
- `NORTH_STAR.md`
- `pyproject.toml`
- `uv.lock`
- `configs/**`
- `data/README.md`
- `docs/**`
- `kgpt/**`
- `tokenizer/**`
- `train/**`
- `eval/**`
- `inference/**`
- `experiments/README.md`
- `tests/**`
- `PROGRESS.md`

## Forbidden Paths

These paths are forbidden to commit or manually edit as phase source changes. Required commands may generate ignored evidence artifacts only when this phase's artifact policy allows them.

- `.env`
- `.env.*`
- `.venv/**`
- `data/raw/**`
- `data/processed/**`
- `data/tokenized/**`
- `experiments/runs/**`
- `*.pt`
- `*.safetensors`
- `agentic-phase-runner-package/**`
- `automation/**`
- `phase-plans/**`
- `concept-and-ideas/**`
- `runs/**`

## Tasks

- Create the Python package skeleton with minimal, importable modules.
- Add a config format and at least one dummy config under `configs/`.
- Define the initial config schema with at least: run name, seed, device, dtype, model name, vocab size, context length, batch size, train steps or epochs, optimizer settings, output directory, checkpoint cadence, eval cadence, and sample cadence.
- Add deterministic seeding helpers for Python, NumPy, and PyTorch.
- Add checkpoint save/load helpers with tests and metadata fields: schema version, config hash or config copy path, model name, step, seed, git commit when available, created timestamp, metrics snapshot, and tokenizer identifier when applicable.
- Add a dummy training entrypoint that trains for one epoch on fake token data and writes `config`, `metrics.jsonl`, `samples.txt`, `checkpoint_last.pt`, and a run manifest under `experiments/runs/`.
- Add initial CLI entrypoints for dummy training and future generation/evaluation discovery, using `python -m ...` or documented console scripts.
- Add an optional MPS smoke command that reports availability and can be skipped when MPS is unavailable.
- Add docs for local setup, validation commands, data conventions, and experiment artifact conventions.
- Update `PROGRESS.md` with completed work, validation results, and known gaps.

## Deliverables

- Python package skeleton for `kgpt`, `train`, `eval`, `inference`, and `tokenizer`.
- `pyproject.toml` and `uv.lock`.
- Dummy config under `configs/`.
- Dummy training command and checkpoint helpers.
- Tests for config loading, deterministic seed behavior, checkpoint roundtrip, and dummy training artifact creation.
- Root docs for setup, data, experiments, and validation.

## Evidence Artifacts

- Ignored dummy run directory under `experiments/runs/{timestamp}_{run_name}/`.
- Run manifest listing command, config path, seed, device, output files, and validation status.
- `metrics.jsonl` with at least: step, tokens seen, train loss, validation loss when available, perplexity when available, learning rate, gradient norm when available, tokens/sec when available, memory usage when available, and generated sample text or sample path.
- Checkpoint metadata for `checkpoint_last.pt`.
- `PROGRESS.md` validation log.

## Artifact Policy

- The dummy run directory, checkpoint files, generated samples, and local logs are allowed to be generated locally but must remain ignored.
- Commit only code, configs, docs, tests, and small fixtures.
- No real corpora or private data may be added in this phase.

## Acceptance Criteria

- `uv run pytest` passes.
- `uv run ruff check .` passes.
- `git diff --check` passes.
- Dummy training runs from config without private data.
- Checkpoint save/load works in tests.
- Generated data, checkpoints, and run artifacts remain ignored by git.
- README explains how to build the runner, inspect status, and run the first dry-run.
- The run directory schema matches the North Star training artifact shape closely enough for later phases to reuse without redesign.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`
- Optional when MPS is present: documented MPS smoke command reports device availability without being required for CI-like success.
- Add the exact dummy training smoke command here once implemented, and require it before marking PHASE-00B complete.

## Human Decisions

- Confirm Python version target if local machine differs from Python 3.11/3.12.
- Confirm whether `argparse` or Typer becomes the CLI convention before adding broad command surfaces; default to `argparse` unless a later phase chooses Typer.

## Phase Gate

Mark complete only when the dummy training command can create a reproducible ignored run directory, tests pass, docs describe the workflow, and no generated artifacts are staged.

## Risks

- The foundation can become too abstract if there is no runnable dummy training command.
- PyTorch/MPS dependencies should not be required for CI-like validation beyond importability and dummy CPU tests.

## Out Of Scope

- Character language modeling.
- Real tokenizer training.
- Real Transformer implementation.
- Real dataset collection.
- Long training runs.

## Deferred Backlog

- Real fixed evaluation prompt set begins in PHASE-01A and expands through later phases.
- Serious tokenizer and data provenance manifests begin in PHASE-02A.
