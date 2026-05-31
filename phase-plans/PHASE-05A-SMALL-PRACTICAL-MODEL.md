# PHASE-05A - Small Practical Model

## Goal

Train a model large enough to be educationally meaningful, targeting roughly 30M to 100M parameters as hardware allows.

## Scope

- Bilingual data mixture decisions.
- Larger config ladder such as `kgpt-30m`, `kgpt-50m`, and optional `kgpt-100m`.
- Throughput and memory profiling on Mac.
- Checkpoint comparison and generation snapshots.

## Prerequisites

- PHASE-04A is complete with a successful tiny pretraining report.
- Final tokenizer choice for 30M+ training is approved or explicitly documented as provisional.
- Dataset sources, licenses, split, dedup, leakage checks, and mixture ratios are documented.

## Phase Dependencies

- Depends on PHASE-04A tiny pretraining evidence and PHASE-02A tokenizer/data provenance.
- Unblocks PHASE-06A, PHASE-07A, and PHASE-08A by producing the required serious base checkpoint and scaling evidence.

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

These paths are forbidden to commit or manually edit as phase source changes. Required commands may generate ignored evidence artifacts only when this phase's artifact policy allows them.

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
- Add concrete configs for `kgpt-30m` and stretch configs for `kgpt-50m` and `kgpt-100m`, each with computed parameter count.
- Choose and document final tokenizer for 30M+ training based on PHASE-02A and PHASE-04A evidence.
- Define data mixture ratios before training, defaulting to a documented English/Japanese mix from the North Star unless human or unattended-policy approval chooses otherwise.
- Add profiling and reporting for Mac memory and throughput.
- Train at least one 30M+ model from random initialization unless infeasible; infeasibility requires evidence, a smaller fallback run, and a human or unattended-policy approved explanation.
- Attempt `kgpt-50m` or `kgpt-100m` only after `kgpt-30m` passes the scale gate, or document why the stretch run is not feasible.
- Document bottlenecks and scaling behavior.

## Deliverables

- `kgpt-30m` config and at least one stretch config.
- Final tokenizer decision note.
- Data mixture manifest with ratios and sources.
- Throughput/memory profiling command and report.
- 30M+ training report or human or unattended-policy approved infeasibility report with fallback evidence.

## Evidence Artifacts

- Ignored run directory for the 30M+ run with config, metrics, samples, checkpoints, tokenizer_info, eval report, and manifest.
- Scaling comparison report covering micro, tiny, and 30M+ behavior.
- Mac profiling report with device, dtype, context length, batch size, tokens/sec, peak memory where measurable, and bottlenecks.

## Artifact Policy

- Large checkpoints, datasets, tokenized files, benchmark logs, and generated samples remain ignored.
- Commit configs and summary reports; do not commit model weights.

## Acceptance Criteria

- At least one 30M+ model is trained from scratch to a documented completion point with validation loss, fixed-prompt samples, and checkpoint metadata, unless infeasible with documented human or unattended-policy approved fallback.
- Loss curves and generation snapshots are saved outside git and summarized in docs.
- Training bottlenecks are documented.
- The report compares behavior to smaller runs.
- Final tokenizer and data mixture choices are documented.
- Stretch `kgpt-50m`/`kgpt-100m` decision is recorded with evidence.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`
- `uv run python -m train.pretrain --config configs/kgpt_30m.yaml --dry-run --validate-resume`
- `uv run python -m eval.compare_runs --manifest docs/phase05a_scaling_manifest.json --output docs/phase05a_scaling_report.md`

## Human Decisions

- Approve final tokenizer and data mixture before 30M+ training.
- Approve fallback if 30M+ training is infeasible on available hardware/time.
- Approve whether to attempt `kgpt-50m` or `kgpt-100m`.

## Phase Gate

Mark complete only when the Definition of Done requirement for at least one 30M+ trained model is satisfied, or when a documented compute limit and fallback is explicitly accepted by the human/orchestrator or unattended decision policy.

## Risks

- Compute time may dominate; reduce scale rather than weakening evidence.

## Out Of Scope

- Production chatbot claims.
- RLHF or advanced alignment.

## Deferred Backlog

- Instruction behavior is deferred to PHASE-06A.
- Full cross-checkpoint evaluation expansion is deferred to PHASE-07A.
