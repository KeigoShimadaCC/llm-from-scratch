# PHASE-10A - Corpus Source Registry

## Goal

Define the allowed real-corpus sources for the next training track and make their provenance, license status, attribution requirements, and local storage policy auditable before any download or cleaning work begins.

## Prerequisites

- PHASE-09A is complete and the final write-up honestly labels the original North Star as partially achieved with documented fallback.
- `docs/NORTH_STAR_LLM_FROM_SCRATCH_MAC.md` data, tokenizer, and training sections have been re-read.
- `automation/policies/unattended-decisions.json` allows the selected source scope.

## Scope

- Source allowlist for `corpus_v01`:
  - English Wikipedia dump (`enwiki`) as the primary English corpus.
  - Japanese Wikipedia dump (`jawiki`) as the primary Japanese corpus.
  - Project Gutenberg as an English public-domain prose supplement.
  - Aozora Bunko as a Japanese public-domain prose supplement.
- `configs/corpus_v01.yaml` schema for source ids, URLs or URL templates, expected license/terms notes, language tags, storage paths, size limits, smoke limits, attribution fields, and exclusion rules.
- `corpus.audit_sources` CLI that checks source entries without downloading corpus payloads.
- Committed source manifest report at `docs/corpus_v01_source_manifest.md`.
- CI-safe tests for config loading and source audit behavior.

## Allowed Paths

- `configs/**`
- `corpus/**`
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

- Depends on PHASE-09A.
- Unblocks PHASE-10B download and cleaning.
- A source is usable downstream only if this phase records license/provenance/attribution evidence and the audit command passes.

## Tasks

- Add `configs/corpus_v01.yaml` with the selected source allowlist and no private or credentialed sources.
- Implement `corpus.audit_sources` with a dry, deterministic audit that validates required metadata fields, storage paths, allowed source ids, language tags, checksum policy, terms/license notes, and attribution requirements.
- Mark Tatoeba as explicitly deferred for future evaluation or SFT helper data, not part of `corpus_v01` pretraining.
- Document local storage policy for raw, processed, and tokenized artifacts under ignored `data/**` directories.
- Add tests that fail if a source lacks license/provenance/attribution metadata or if an unapproved source is added without policy update.
- Generate `docs/corpus_v01_source_manifest.md` from the audit CLI.

## Deliverables

- `configs/corpus_v01.yaml`
- `corpus/` package with source registry and audit CLI.
- `docs/corpus_v01_source_manifest.md`
- Tests covering source metadata validation and allowlist enforcement.
- `PROGRESS.md` update naming decisions, validation, and any blocked source.

## Evidence Artifacts

- Committed source manifest at `docs/corpus_v01_source_manifest.md`.
- Optional ignored runner evidence under `runs/phase-runner/PHASE-10A/**`.
- No raw corpus data is required or allowed for this phase.

## Acceptance Criteria

- The selected public sources are explicit and bounded to English Wikipedia, Japanese Wikipedia, Project Gutenberg, and Aozora Bunko.
- Every source has source id, language, provenance URL or URL template, license/terms note, attribution requirement, local storage path, and exclusion/blocker policy.
- Any source whose terms cannot be verified is excluded or marked blocked and is not eligible for PHASE-10B.
- The audit command produces the committed manifest without downloading raw corpus payloads.
- No OpenAI embeddings, pretrained model weights, or pretrained model wrappers are introduced.

## Required Validation

- `uv run pytest`
- `uv run ruff check .`
- `git diff --check`
- `uv run python -m corpus.audit_sources --config configs/corpus_v01.yaml --output docs/corpus_v01_source_manifest.md`

## Artifact Policy

- Raw, processed, tokenized, tokenizer model, checkpoint, and runner evidence artifacts remain ignored.
- This phase must not download corpus payloads.
- The committed manifest may include source metadata, URLs, checksums when known, attribution instructions, and blocker notes, but not full corpus text.

## Human Decisions

- Approve the selected `corpus_v01` source allowlist.
- Confirm whether license/terms notes are sufficient for unattended download and cleaning.
- Decide whether any uncertain source should be blocked instead of included.

## Phase Gate

Mark complete only when `configs/corpus_v01.yaml`, `corpus.audit_sources`, committed source manifest, tests, and required validation pass, and every included source is backed by documented provenance, license/terms, and attribution metadata.

## Risks

- Public access does not automatically mean training use is acceptable; unclear terms must block a source.
- Source URLs and dump naming conventions may drift over time, so the audit must distinguish config validity from live download success.

## Out Of Scope

- Downloading, extracting, cleaning, splitting, tokenizing, or training on corpus text.
- Adding Tatoeba to `corpus_v01`.
- Implementing embeddings or retrieval.

## Deferred Backlog

- PHASE-11B may add evaluation helper datasets after real-corpus pretraining exists.
- PHASE-12A may add better instruction data after the base model is trained.
