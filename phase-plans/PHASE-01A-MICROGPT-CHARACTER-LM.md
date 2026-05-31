# PHASE-01A - MicroGPT Character LM

## Goal

Build the smallest complete language-modeling loop using character-level data.

## Scope

- Character vocabulary and encode/decode utilities.
- Tiny baseline and/or tiny Transformer path.
- Next-character prediction.
- Greedy and sampling-based generation.
- Validation loss and a tiny overfit test.
- Short explanation of underfitting, overfitting, and sampling temperature.

## Prerequisites

- PHASE-00B is complete.
- The dummy training command, checkpoint helpers, config loader, and run artifact conventions exist.
- The fixed evaluation prompt set can be started in this phase even if it is small.

## Phase Dependencies

- Depends on PHASE-00B foundation artifacts.
- Unblocks PHASE-02A by proving the training loop, deterministic generation, and first reusable eval prompts.

## Allowed Paths

- `configs/**`
- `data/README.md`
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

- Add character tokenizer utilities and tests.
- Add a small public fixture corpus for tests only, with source and license documented in the test fixture or docs. Default to repo-authored synthetic text if no public-domain corpus is chosen.
- Add a micro training path that can overfit tiny text.
- Add sample generation with deterministic seeding and temperature.
- Implement either a bigram baseline first or a tiny Transformer first, and document the decision. Default to bigram baseline plus a minimal Transformer only if time allows.
- Add deterministic generation checks for greedy decoding and seeded sampling.
- Document the first training curve and sample behavior in a mini report.

## Deliverables

- Character tokenizer/vocabulary implementation.
- Tiny corpus fixture with source/license note.
- Micro training config and command.
- Seeded generation path.
- Mini report under `docs/`.

## Evidence Artifacts

- Ignored run directory with config, metrics, samples, checkpoint, and manifest.
- Mini report with: corpus source/license, model type, parameter count, config, initial/final loss, overfit threshold result, sample prompts, generated samples, and known failure modes.
- Fixed prompt file or documented prompt list to reuse in later phases.

## Artifact Policy

- Generated checkpoints, samples, and run directories may be created locally but must not be committed.
- Commit only small source files, configs, tests, reports, and tiny fixtures with documented provenance.

## Acceptance Criteria

- Character tokenizer roundtrip tests pass.
- Training loss decreases on the tiny corpus.
- The model can overfit a tiny dataset to a documented threshold, defaulting to train loss below 0.1 or a clearly justified phase-specific threshold.
- Generation produces non-empty text and supports deterministic seeds.
- A short experiment note exists under `docs/`.
- Greedy generation is deterministic for a fixed checkpoint and prompt.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`
- Add the exact micro overfit training command and deterministic generation check command here once implemented, and require them before marking PHASE-01A complete.

## Human Decisions

- Approve the corpus source/license if anything other than repo-authored synthetic text is used.
- Decide whether the baseline-only result is enough to complete the phase or whether the tiny Transformer must be included before moving to tokenization.

## Phase Gate

Mark complete only when a tiny character model demonstrably overfits, deterministic generation is tested, and the mini report explains underfitting, overfitting, and sampling temperature.

## Risks

- A model that cannot overfit tiny text is not ready for token-level work.

## Out Of Scope

- BPE or SentencePiece tokenizer work.
- 5M+ parameter training.
- Instruction tuning.

## Deferred Backlog

- Expand the fixed prompt set in tokenizer and Transformer phases.
- Add richer generation metrics in PHASE-07A.
