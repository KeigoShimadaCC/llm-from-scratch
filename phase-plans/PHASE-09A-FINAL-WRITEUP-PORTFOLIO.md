# PHASE-09A - Final Write-Up And Portfolio Layer

## Goal

Make the project legible to technical reviewers as a serious educational research-engineering artifact.

## Scope

- Final technical write-up.
- Architecture, tokenizer, data, training, evaluation, failure analysis, and Mac optimization summaries.
- Clear limitations and lessons learned.
- README polish.

## Prerequisites

- All prior phases are complete or explicitly marked with accepted blockers.
- Serious run reports, tokenizer reports, data manifests, eval reports, inference benchmarks, and model cards exist.
- Claims are backed by committed summaries and ignored artifact manifests.

## Phase Dependencies

- Depends on all prior phase gates or accepted blockers.
- Closes the original North Star only when every major claim maps to committed evidence and reproducible commands without accepted fallback replacing a required outcome.
- If 30M+ training or MLX inference is formally deferred, this phase may close a pragmatic project milestone but must not label the original North Star as fully achieved.

## Allowed Paths

- `README.md`
- `NORTH_STAR.md`
- `docs/**`
- `eval/**`
- `inference/**`
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

- Write the final report.
- Add clear setup, training, evaluation, and inference instructions.
- Summarize what was learned from failures and scaling limits.
- Make the repo navigable for an engineer who already knows LLMs.
- Add a claim-to-evidence table mapping every major claim to a report, command, config, run manifest, or artifact path.
- Add artifact index for ignored run directories and committed reports.
- Add reproducible command index for setup, training, evaluation, inference, and benchmark flows.
- Add or link model cards and data cards.
- Add an explicit "what changed between phases" section.
- Add explicit final positioning: `North Star achieved`, `North Star partially achieved with documented fallback`, or `North Star blocked`, with evidence for the label.

## Deliverables

- Final technical write-up.
- Updated README navigation.
- Claim-to-evidence table.
- Artifact index.
- Reproducible command index.
- Model/data cards or links to them.

## Evidence Artifacts

- Links or paths to run reports, tokenizer reports, data manifests, eval reports, inference benchmark reports, and model cards.
- Table covering architecture, tokenizer, data, training dynamics, scaling, SFT, evaluation, Mac inference, limitations, and lessons learned.

## Artifact Policy

- Final docs should point to ignored artifacts by path and manifest but must not commit checkpoints, corpora, tokenized data, or full generated run directories.

## Acceptance Criteria

- Final write-up explains architecture, experiments, limitations, and Mac-specific decisions.
- README points to the most important docs and commands.
- Claims are supported by reports and validation evidence.
- Claim-to-evidence table exists and has no unsupported major claim.
- Reproducible command index is complete enough for a fresh engineer to rerun key flows.
- "What changed between phases" section exists.
- If 30M+ training or MLX inference was deferred, the final write-up says the original North Star was not fully achieved and explains the accepted fallback.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`
- `uv run python -m eval.audit_claims --doc docs/FINAL_WRITEUP.md --output docs/claim_evidence_audit.md`
- `uv run python -m eval.check_repro_commands --doc docs/COMMAND_INDEX.md`

## Human Decisions

- Approve final claims and wording before publishing.
- Approve any unresolved blockers or reduced-scope outcomes.

## Phase Gate

Mark complete only when the final write-up is evidence-backed, reproducible command paths are documented, unsupported claims have been removed or labeled as future work, and the final positioning honestly distinguishes full North Star completion from documented fallback.

## Risks

- Portfolio polish must not overstate model capability.

## Out Of Scope

- New model training work unless required to fix a documentation claim.

## Deferred Backlog

- Future portfolio polish can improve presentation, but not by weakening evidence requirements.
