# PHASE-05A - Small Practical Model

## Goal

Train a model large enough to be educationally meaningful, targeting roughly 30M to 100M parameters as hardware allows.

## Scope

- Bilingual data mixture decisions.
- Larger config ladder such as `kgpt-30m`, `kgpt-50m`, and optional `kgpt-100m`.
- Throughput and memory profiling on Mac.
- Checkpoint comparison and generation snapshots.

## Allowed Paths

- `configs/**`
- `docs/**`
- `kgpt/**`
- `train/**`
- `eval/**`
- `inference/**`
- `tests/**`
- `PROGRESS.md`

## Forbidden Paths

- `.env`
- `.env.*`
- `data/raw/**`
- `data/processed/**`
- `data/tokenized/**`
- `experiments/runs/**`
- `*.pt`
- `*.safetensors`
- `agentic-phase-runner-package/**`
- `automation/**`
- `phase-plans/**`
- `runs/**`

## Tasks

- Define the small model config ladder.
- Add profiling and reporting for Mac memory and throughput.
- Run at least one 30M+ training attempt if feasible.
- Document bottlenecks and scaling behavior.

## Acceptance Criteria

- At least one 30M+ model run is attempted from scratch.
- Loss curves and generation snapshots are saved outside git and summarized in docs.
- Training bottlenecks are documented.
- The report compares behavior to smaller runs.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`

## Risks

- Compute time may dominate; reduce scale rather than weakening evidence.

## Out Of Scope

- Production chatbot claims.
- RLHF or advanced alignment.
