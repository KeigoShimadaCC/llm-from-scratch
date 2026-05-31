# PHASE-02A - Tokenizer And Dataset Pipeline

## Goal

Move from character-level modeling to real token-level LLM training.

## Scope

- Train or configure a small tokenizer for English/Japanese experiments.
- Compare English-only and bilingual Japanese/English tokenization.
- Implement train/validation split generation.
- Add chunked or memory-mapped token files.
- Add batch sampling for language-model training.
- Report tokenizer statistics and failure cases.

## Prerequisites

- PHASE-01A is complete and the character-level loop can overfit tiny data.
- The repository has a documented artifact policy for ignored local data and tokenized files.
- Human approval is available for dataset source/license decisions.

## Phase Dependencies

- Depends on PHASE-01A training-loop and prompt assets.
- Unblocks PHASE-03A and PHASE-04A by producing token batches, tokenizer reports, split metadata, and data provenance.

## Allowed Paths

- `configs/**`
- `data/README.md`
- `docs/**`
- `kgpt/**`
- `tokenizer/**`
- `train/**`
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

- Add tokenizer training or configuration scripts for at least byte-level BPE and SentencePiece unigram/BPE candidates, unless a candidate is explicitly rejected with rationale.
- Run a vocabulary-size sweep for candidate sizes relevant to the North Star: 8k, 16k, and 32k where feasible; document if a size is skipped.
- Add tokenizer inspection and report generation.
- Add dataset source manifest fields: source name, URL or local note, license, language mix, size, checksum when available, preprocessing command, split method, dedup strategy, and contamination/leakage notes.
- Add dataset preprocessing, dedup, train/validation split generation, token file writing, and batch sampling.
- Define token file format, dtype, metadata sidecar, tokenizer identifier, and split names.
- Add roundtrip, split-integrity, leakage-check, dedup-smoke, and batch-shape tests.
- Document English/Japanese tokenization tradeoffs.

## Deliverables

- Tokenizer training/config scripts.
- Tokenizer inspection/report script.
- Dataset manifest format.
- Tokenized split metadata format.
- Batch sampler for token-level LM training.
- Tokenizer report under `docs/`.

## Evidence Artifacts

- Tokenizer report with vocabulary size, English tokens per sentence, Japanese tokens per sentence, unknown-token behavior, byte fallback behavior, bad segmentation examples, compression ratio, and effect on context length.
- Data manifest documenting provenance, license, split, dedup, and leakage checks.
- Ignored tokenizer model files and tokenized data files, with paths listed in the report.

## Artifact Policy

- Raw, processed, and tokenized data may be generated locally under ignored `data/**` paths but must not be committed.
- Tokenizer model files may remain ignored if large or data-derived; commit only small configs/reports unless the phase explicitly justifies committing a tiny tokenizer artifact.

## Acceptance Criteria

- Tokenizer encode/decode roundtrip tests pass.
- Dataset batching returns expected tensor shapes.
- Tokenizer report includes vocabulary size, English/Japanese token-per-sentence stats, unknown/byte behavior, compression, and bad segmentation examples.
- Data source manifest exists with source/license/split/dedup/leakage notes.
- Token file format and metadata are documented.
- A small model can consume token batches.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`
- Add the exact tokenizer report generation, data manifest validation, and token batch sampler smoke commands here once implemented, and require them before marking PHASE-02A complete.

## Human Decisions

- Approve dataset sources and licensing before they are used for training claims.
- Choose the tokenizer candidate to carry forward to PHASE-04A based on the report, or explicitly defer final choice with rationale.

## Phase Gate

Mark complete only when tokenized train/validation batches are reproducible from documented sources, tokenizer quality is measured for English and Japanese, and leakage/dedup checks are documented.

## Risks

- Large or private datasets must remain outside git.
- Oversized vocabularies can waste parameters in small models.

## Out Of Scope

- Long pretraining runs.
- Final main-model tokenizer decision if evidence is insufficient.

## Deferred Backlog

- Final tokenizer choice may be revisited in PHASE-05A before 30M+ training.
- Larger data stages are deferred until pretraining phases.
