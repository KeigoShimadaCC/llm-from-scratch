# PHASE-11A - Meaningful 30M Training

## Goal

Run the first meaningful `kgpt-30m` training track on the real `corpus_v01` tokenized dataset and produce evidence-backed reports, checkpoint comparisons, and an inference smoke path without claiming quality beyond the data.

## Prerequisites

- PHASE-10D is complete.
- `configs/corpus_v01_tokenized.yaml` can produce smoke batches.
- `docs/tokenizer_corpus_v01_report.md` records the final tokenizer choice.
- Ignored tokenized arrays exist locally.
- PHASE-05A 30M architecture and training loop evidence remain valid.

## Scope

- `configs/kgpt_30m_corpus_v01.yaml` for real-corpus `kgpt-30m` training.
- Dry-run and resume validation before training.
- Smoke-sized but meaningful local training run named `phase11a_kgpt30m_corpus_v01_smoke`.
- Checkpoint manifest at `docs/checkpoint_manifest_corpus_v01.json`.
- Comparison report at `docs/phase11a_real_corpus_checkpoint_comparison.md`.
- Fixed-prompt sample snapshots and loss/perplexity/throughput evidence for the real-corpus run.

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

Forbidden paths are forbidden to commit or manually edit as phase source changes. Required commands may generate ignored evidence artifacts only when this plan's artifact policy allows them.

- `.env`
- `.env.*`
- `data/raw/**`
- `data/processed/**`
- `data/tokenized/**`
- `experiments/runs/**`
- `runs/**`
- `*.pt`
- `*.safetensors`
- `corpus/**`
- `tokenizer/**`
- `automation/**`
- `phase-plans/**`

## Phase Dependencies

- Depends on PHASE-10D.
- Unblocks PHASE-11B evaluation expansion and PHASE-12A better SFT.

## Tasks

- Add `configs/kgpt_30m_corpus_v01.yaml` pointing at the selected tokenizer and `corpus_v01` tokenized dataset.
- Ensure dry-run validation checks config compatibility, parameter count, data paths, checkpoint metadata, and resume behavior without training.
- Run `phase11a_kgpt30m_corpus_v01_smoke` for 1000 steps unless compute or data evidence proves infeasibility.
- Record metrics: step, tokens seen, train loss, validation loss, perplexity, learning rate, gradient norm, tokens/sec, memory usage when available, and generated samples.
- Add or update checkpoint comparison manifest and report for real-corpus checkpoints.
- Ensure fixed-prompt samples include English and Japanese continuations and are explicitly labeled as educational quality probes, not chatbot claims.
- Add tests for new config parsing, checkpoint manifest compatibility, and report rendering where practical.
- Update `PROGRESS.md` with ignored run directory, checkpoint paths, validation, and any fallback decision.

## Deliverables

- `configs/kgpt_30m_corpus_v01.yaml`
- `docs/checkpoint_manifest_corpus_v01.json`
- `docs/phase11a_real_corpus_checkpoint_comparison.md`
- Training/eval/report code updates only as needed for real-corpus compatibility.
- Tests covering config/report behavior.
- Ignored run directory under `experiments/runs/phase11a_kgpt30m_corpus_v01_smoke/**`.
- `PROGRESS.md` update with metrics and limitations.

## Evidence Artifacts

- Ignored run directory with config, metrics, samples, checkpoint metadata, checkpoint files, tokenizer info, and eval report.
- Committed checkpoint manifest and comparison report.
- Optional ignored runner evidence under `runs/phase-runner/PHASE-11A/**`.

## Acceptance Criteria

- `kgpt-30m` real-corpus config dry-run and resume validation pass.
- A 1000-step `phase11a_kgpt30m_corpus_v01_smoke` run completes unless infeasibility is documented with command evidence, smaller fallback, and final positioning constraints.
- Reported evidence includes loss/perplexity trend, sample snapshots, tokenizer/data references, checkpoint metadata, throughput or memory notes, and limitations.
- The report compares real-corpus checkpoints using the fixed evaluation prompt set.
- Generated checkpoints and run directories remain ignored.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`
- `uv run python -m train.pretrain --config configs/kgpt_30m_corpus_v01.yaml --dry-run --validate-resume`
- `uv run python -m train.pretrain --config configs/kgpt_30m_corpus_v01.yaml --max-steps 1000 --run-name phase11a_kgpt30m_corpus_v01_smoke`
- `uv run python -m eval.compare_checkpoints --manifest docs/checkpoint_manifest_corpus_v01.json --output docs/phase11a_real_corpus_checkpoint_comparison.md`

## Artifact Policy

- Training checkpoints, run metrics, sample logs, tokenized arrays, and processed corpus artifacts remain ignored.
- Committed reports must name run paths, commands, config hashes, data/tokenizer manifests, metrics, and limitations without embedding checkpoint binaries or corpus text.
- Do not commit `experiments/runs/**`, `data/**` generated payloads, `*.pt`, or `*.safetensors`.

## Human Decisions

- Approve the run budget and stop/continue policy if the 1000-step run is too slow or unstable.
- Decide whether loss/sample evidence is enough to call the run meaningful or whether it remains a smoke-only fallback.
- Approve any fallback wording if compute or data readiness blocks the 1000-step run.

## Phase Gate

Mark complete only when dry-run/resume validation, the required training run or documented fallback, checkpoint comparison report, tests, lint, whitespace, and artifact policy checks pass, and `PROGRESS.md` records exact run evidence and limitations.

## Risks

- A 1000-step 30M run may still be too small to show coherent generation; the report must avoid overstated capability claims.
- Local Mac memory or throughput may force reduced batch/context settings; any reduction must be config-driven and documented.

## Out Of Scope

- Full long-form pretraining to convergence.
- MLX implementation.
- Better instruction tuning.
- Broad benchmark expansion.

## Deferred Backlog

- PHASE-11B: expand evaluation after real-corpus checkpoints exist.
- PHASE-12A: build a stronger SFT dataset and instruction tuning pass from the real-corpus base model.
