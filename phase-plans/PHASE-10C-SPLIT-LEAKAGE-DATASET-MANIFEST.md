# PHASE-10C - Split, Leakage, And Dataset Manifest

## Goal

Create deterministic document-level train/validation/test split and leakage checks for the cleaned `corpus_v01` smoke corpus, with committed manifest summaries that are safe to review and reproduce.

## Prerequisites

- PHASE-10B is complete.
- Ignored smoke processed corpus exists at `data/processed/corpus_v01_smoke`.
- `configs/corpus_v01.yaml` still passes PHASE-10A source audit.

## Scope

- `corpus.split_manifest` CLI for deterministic split, exact dedup, leakage checks, and committed summary output.
- Document-level split policy for train/validation/test.
- Exact normalized-text hash deduplication before split.
- Leakage checks across split hashes and source record ids.
- Dataset manifest summary at `docs/corpus_v01_dataset_manifest.json`.

## Allowed Paths

- `configs/**`
- `corpus/**`
- `data/README.md`
- `docs/**`
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
- `kgpt/**`
- `tokenizer/**`
- `train/**`
- `eval/**`
- `inference/**`
- `automation/**`
- `phase-plans/**`

## Phase Dependencies

- Depends on PHASE-10B.
- Unblocks PHASE-10D tokenizer and tokenized dataset creation.

## Tasks

- Implement deterministic document split using a stable seed and normalized document hash, not line order.
- Add exact dedup before split using normalized text hashes and source record id collision checks.
- Define manifest schema with source counts, language counts, byte and character totals, document counts, split ratios, dedup counts, excluded counts, leakage probe results, config hash, and generated timestamp.
- Ensure committed manifest summaries do not include full document text.
- Add tests for split determinism, hash stability, duplicate removal, leakage failure, and manifest schema.
- Update `data/README.md` with split and manifest conventions.

## Deliverables

- `corpus.split_manifest` CLI.
- `docs/corpus_v01_dataset_manifest.json`.
- Tests for split, dedup, leakage, and manifest schema.
- `PROGRESS.md` update with local input/output artifact paths and validation commands.

## Evidence Artifacts

- Committed dataset manifest summary at `docs/corpus_v01_dataset_manifest.json`.
- Ignored input artifacts from PHASE-10B under `data/processed/corpus_v01_smoke/**`.
- Optional ignored split sidecars under `data/processed/**` only if needed by downstream tooling.

## Acceptance Criteria

- Train/validation/test membership is deterministic and document-level.
- Exact duplicate normalized documents cannot appear across splits.
- Leakage checks fail loudly on hash collisions across train/validation/test.
- Manifest summary is committed, reviewable, and free of full corpus text.
- Source and language mixture counts are explicit enough to guide tokenizer and training decisions.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`
- `uv run python -m corpus.split_manifest --config configs/corpus_v01.yaml --processed data/processed/corpus_v01_smoke --output docs/corpus_v01_dataset_manifest.json`

## Artifact Policy

- Processed corpus and optional split sidecars remain ignored.
- Committed manifest contains aggregate metadata, hashes, counts, and provenance references only.
- Do not commit raw, processed, or tokenized corpus records.

## Human Decisions

- Approve split ratios and source/language mixture for `corpus_v01`.
- Decide whether exact dedup is sufficient for this phase or whether near-dedup must block PHASE-10D.
- Approve any excluded source subsets if quality or license filters are broad.

## Phase Gate

Mark complete only when the split manifest command passes on the PHASE-10B smoke corpus, leakage checks pass, manifest schema is tested, and ignored artifact policy is verified.

## Risks

- Exact dedup will not catch paraphrases or boilerplate variants; the limitation must be documented in the manifest.
- Smoke corpus results may not represent the full local corpus mixture.

## Out Of Scope

- Near-duplicate detection beyond exact normalized hashes.
- Tokenizer training.
- Model training.

## Deferred Backlog

- Add near-dedup or MinHash-style checks after full local corpus processing if exact dedup misses obvious boilerplate repeats.
