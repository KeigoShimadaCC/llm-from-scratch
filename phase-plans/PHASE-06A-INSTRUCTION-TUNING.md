# PHASE-06A - Instruction Tuning Layer

## Goal

Convert a pretrained scratch model into a simple instruction-following variant.

## Scope

- Small instruction dataset.
- Supervised fine-tuning loop.
- Prompt format and evaluation probes.
- Base-vs-SFT comparison.

## Prerequisites

- PHASE-05A is complete or a human-approved fallback base checkpoint exists.
- Base checkpoint, tokenizer, and config are compatible with SFT loading.
- Instruction dataset source/license is approved.

## Phase Dependencies

- Depends on PHASE-05A base checkpoint or accepted fallback plus tokenizer/config compatibility.
- Unblocks PHASE-07A instruction-tuned comparison and PHASE-09A final instruction-tuning claims.

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

- Define the instruction prompt format.
- Version the prompt template and store the version in config and checkpoint metadata.
- Add instruction dataset manifest with source, license, size, split, dedup, and contamination notes.
- Add SFT data loading and training.
- Implement prompt-vs-response loss masking so only the intended target span contributes when configured.
- Add train/validation split and fixed SFT eval set.
- Add base-vs-SFT fixed prompt comparisons.
- Document what the model can and cannot follow.

## Deliverables

- SFT config and prompt template version.
- Instruction data manifest.
- SFT training path with optional response-only loss masking.
- Base-vs-SFT evaluation report.

## Evidence Artifacts

- Ignored SFT run directory with config, metrics, samples, checkpoint, eval report, and manifest.
- Fixed SFT eval set results for base and tuned checkpoints.
- Report documenting dataset source/license, prompt format, masking policy, validation behavior, improvements, and limitations.

## Artifact Policy

- Instruction datasets, checkpoints, and generated run artifacts remain ignored unless a tiny synthetic fixture is explicitly committed for tests.
- Commit prompt template, configs, code, tests, and summary reports.

## Acceptance Criteria

- Instruction-tuned checkpoint exists outside git.
- Base-vs-SFT comparison is documented.
- Narrow instruction-following behavior improves on fixed probes.
- Limitations are clear and not overstated.
- Prompt format version, dataset source/license, split, and masking policy are documented.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`
- Add the exact SFT smoke training, fixed SFT eval, and base-vs-SFT comparison commands here once implemented, and require them before marking PHASE-06A complete.

## Human Decisions

- Approve instruction dataset source/license.
- Approve base checkpoint selection.
- Approve the final prompt format version before serious SFT.

## Phase Gate

Mark complete only when SFT produces a checkpoint, fixed eval shows a narrow improvement over base, and the report clearly states limits without production safety claims.

## Risks

- Small base models may not have enough language ability for robust instruction following.

## Out Of Scope

- Production safety claims.
- RLHF or preference optimization.

## Deferred Backlog

- Broader safety/alignment work remains out of scope unless a later North Star revision adds it.
