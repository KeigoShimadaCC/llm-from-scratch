# PHASE-10D - Tokenizer And Tokenized Dataset

## Goal

Train and compare `corpus_v01` byte-BPE tokenizer candidates, choose a tokenizer for real-corpus training, and build ignored tokenized arrays that the training sampler can read.

## Prerequisites

- PHASE-10C is complete.
- `docs/corpus_v01_dataset_manifest.json` exists and passes split/leakage checks.
- Ignored processed smoke corpus exists locally for validation.

## Scope

- `configs/tokenizer_corpus_v01.yaml` for `2k`, `4k`, `8k`, and `16k` byte-BPE candidates.
- `configs/corpus_v01_tokenized.yaml` for tokenized dataset reading and batch sampling.
- Tokenizer training/report flow for `corpus_v01`.
- Tokenized array generation under ignored `data/tokenized/**`.
- Batch sampler validation against the chosen tokenizer and token arrays.

## Allowed Paths

- `configs/**`
- `corpus/**`
- `data/README.md`
- `docs/**`
- `kgpt/**`
- `tokenizer/**`
- `train/**`
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
- `eval/**`
- `inference/**`
- `automation/**`
- `phase-plans/**`

## Phase Dependencies

- Depends on PHASE-10C.
- Unblocks PHASE-11A meaningful `kgpt-30m` training on real-corpus tokens.

## Tasks

- Add tokenizer config for `2k`, `4k`, `8k`, and `16k` byte-BPE candidates trained from `corpus_v01` processed text.
- Extend tokenizer/report code as needed to consume the PHASE-10C manifest and processed JSONL records.
- Report vocabulary size, English/Japanese tokens per sentence, unknown/byte fallback behavior, compression ratio, bad segmentation examples, corpus coverage, and model-size tradeoff.
- Define deterministic final tokenizer selection criteria and record the selected candidate in the report.
- Build tokenized train/validation/test arrays under ignored `data/tokenized/**` with metadata sidecars linking source manifest, dataset manifest, tokenizer config, tokenizer hash, and split hashes.
- Ensure `train.sample_batches` can read the generated tokenized config and produce deterministic smoke batches.
- Add tests for tokenizer candidate config, report fields, byte fallback/roundtrip, tokenized sidecar metadata, and sampler compatibility.

## Deliverables

- `configs/tokenizer_corpus_v01.yaml`
- `configs/corpus_v01_tokenized.yaml`
- `docs/tokenizer_corpus_v01_report.md`
- Tokenizer/report code updates.
- Tokenized dataset writing/reading updates if needed.
- Tests for tokenizer and tokenized dataset behavior.
- `PROGRESS.md` update naming the selected tokenizer and ignored tokenized artifact paths.

## Evidence Artifacts

- Committed tokenizer report at `docs/tokenizer_corpus_v01_report.md`.
- Ignored tokenizer model artifacts and token arrays under `data/tokenized/**` or another ignored local data path named by the report.
- Optional ignored runner evidence under `runs/phase-runner/PHASE-10D/**`.

## Acceptance Criteria

- Candidate tokenizers `2k`, `4k`, `8k`, and `16k` are trained or smoke-trained from `corpus_v01` data and compared in one committed report.
- The final tokenizer choice is justified by English/Japanese fragmentation, compression, roundtrip, and embedding-parameter tradeoffs.
- Tokenized arrays are generated locally but not committed.
- Batch sampling against `configs/corpus_v01_tokenized.yaml` succeeds.
- No OpenAI embeddings, pretrained tokenizer model weights, or pretrained language model weights are used in the scratch-core path.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`
- `uv run python -m tokenizer.train_report --config configs/tokenizer_corpus_v01.yaml --output docs/tokenizer_corpus_v01_report.md`
- `uv run python -m train.sample_batches --config configs/corpus_v01_tokenized.yaml --max-batches 2`

## Artifact Policy

- Tokenizer models, tokenized arrays, and large intermediate tokenizer training files are ignored local artifacts.
- Committed reports may include aggregate statistics, examples, paths, hashes, and selected tokenizer metadata, but not full corpus text or token arrays.
- `data/tokenized/**`, `*.pt`, and `*.safetensors` remain untracked.

## Human Decisions

- Approve the selected tokenizer candidate when report metrics trade off English/Japanese quality versus parameter budget.
- Decide whether a poor Japanese segmentation result blocks PHASE-11A or is accepted with documented limitation.

## Phase Gate

Mark complete only when tokenizer report generation, tokenized dataset batch sampling, tests, lint, whitespace, and artifact policy checks pass, and the final tokenizer choice is recorded in committed evidence.

## Risks

- A `16k` tokenizer may consume too much of a 30M model's parameter budget; the report must make this tradeoff explicit.
- Smoke-sized tokenizer evidence may not match full-corpus behavior; the limitation must be documented.

## Out Of Scope

- Model pretraining.
- Instruction tuning.
- Evaluation expansion beyond tokenizer examples and batch sampling.

## Deferred Backlog

- Revisit `32k` or unigram tokenizer candidates only if PHASE-11A evidence shows the byte-BPE choice is a blocker.
