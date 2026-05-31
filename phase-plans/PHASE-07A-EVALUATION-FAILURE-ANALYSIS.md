# PHASE-07A - Evaluation And Failure Analysis

## Goal

Build comparable evaluation for checkpoints and avoid vibes-only assessment.

## Scope

- Loss, perplexity, throughput, memory, and generation metrics.
- Fixed qualitative prompt set.
- Toy task probes.
- Failure taxonomy and report generation.

## Prerequisites

- Earlier phases have produced fixed prompt assets and at least micro/tiny checkpoint reports.
- Serious checkpoints have run manifests and compatible configs.
- Data/tokenizer manifests are available for leakage and memorization analysis.

## Phase Dependencies

- Depends on PHASE-01A through PHASE-06A evidence artifacts and compatible checkpoint metadata.
- Unblocks PHASE-09A by producing comparable evaluation, failure analysis, and evidence tables.

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

- Add evaluation scripts that run against any compatible checkpoint.
- Consolidate fixed prompts introduced earlier for English, Japanese, technical, instruction, bilingual, copy, arithmetic toy, and translation-like probes.
- Define metric calculations for validation loss, perplexity, tokens/sec, memory, repetition rate, average completion length, entropy/top-k behavior, and exact-match toy tasks.
- Add memorization and leakage probes using held-out and near-duplicate checks where data manifests permit.
- Add report generation with failure taxonomy.
- Compare serious model scales in docs.

## Deliverables

- Checkpoint-compatible evaluation CLI.
- Fixed prompt/eval set.
- Report schema for comparable eval reports.
- Failure taxonomy implementation or report template.
- Cross-checkpoint comparison report.

## Evidence Artifacts

- Eval reports for each serious checkpoint.
- Comparison table across micro, tiny, 30M+, SFT, and any stretch checkpoints available.
- Failure analysis covering repetition loops, memorized fragments, bad token boundaries, language mixing, instruction ignored, and false factual confidence.

## Artifact Policy

- Large generated eval outputs may remain ignored if summarized in committed reports.
- Commit prompt sets, small eval fixtures, report schema, and summary reports.

## Acceptance Criteria

- Evaluation can run against any compatible checkpoint.
- Each serious model scale has a comparable report.
- Reports classify failures such as repetition, memorization, language mixing, and tokenization artifacts.
- Metric definitions are documented and applied consistently.
- Earlier fixed prompt assets are reused rather than replaced without rationale.
- Leakage and memorization probes are included or explicitly blocked by missing data evidence.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`
- `uv run python -m eval.report --config configs/eval_fixed_prompts.yaml --output docs/phase07a_eval_report.md`
- `uv run python -m eval.compare_checkpoints --manifest docs/checkpoint_manifest.json --output docs/phase07a_checkpoint_comparison.md`

## Human Decisions

- Approve the fixed eval prompt set before using it for final claims.
- Approve whether a checkpoint is "serious" enough to require a comparable report.

## Phase Gate

Mark complete only when compatible checkpoints can be evaluated with the same prompt/metric schema and the comparison report explains both improvements and failures.

## Risks

- Qualitative samples can mislead without fixed prompts and comparable settings.

## Out Of Scope

- External leaderboard optimization.

## Deferred Backlog

- Any leaderboard or third-party benchmark integration requires a separate phase and contamination review.
