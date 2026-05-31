# Data Card

## Scope

The current project evidence uses repo-authored synthetic fixtures for tokenizer, pretraining, evaluation, and
instruction-tuning smoke tests. No private data, scraped corpora, benchmark answers, or third-party datasets are
committed.

## Data Sources

| Source | Purpose | License | Evidence |
|---|---|---|---|
| `phase02a_synthetic_bilingual_smoke_v1` | English/Japanese tokenizer and token-batch smoke pipeline. | Repo-authored fixture for this project. | `docs/phase02a_data_manifest.json` |
| `phase04a_repo_authored_tiny_pretrain_v1` | Tiny pretraining smoke corpus. | Repo-authored fixture for this project. | `docs/phase04a_data_manifest.json` |
| PHASE-05A data mixture | 30M+ local scale-gate smoke data. | Repo-authored fixture for this project. | `docs/phase05a_data_mixture_manifest.json` |
| `phase06a_repo_authored_instruction_smoke_v1` | Instruction-tuning smoke commands. | Repo-authored fixture for this project. | `docs/phase06a_instruction_data_manifest.json` |
| Fixed eval prompts | Cross-checkpoint qualitative and toy probes. | Repo-owned project prompts. | `configs/eval_fixed_prompts.yaml`, `docs/phase07a_eval_schema.md` |

## Processing

- Normalize with Unicode NFKC where phase manifests specify it.
- Deduplicate by normalized text hashes or normalized instruction/response hashes.
- Split train/validation deterministically with fixed seeds and hash ordering.
- Check exact normalized hash overlap between train and validation where applicable.
- Write token arrays and tokenizer model artifacts to ignored `data/tokenized/` paths.

## Tokenizer Data Notes

The selected tokenizer is a UTF-8 byte-level BPE trained/configured through this repo. It has full byte fallback and
can roundtrip unseen Unicode without relying on unknown tokens, but the tiny corpus leaves Japanese and emoji-heavy
text fragmented.

## Known Data Limitations

- Fixture data is too small for language-quality claims.
- The bilingual mix is useful for pipeline validation, not broad coverage.
- Exact-hash leakage checks do not eliminate semantic similarity across tiny train/validation splits.
- No public benchmark datasets are included, so benchmark contamination is intentionally avoided but benchmark
  competence is not measured.

## Storage Policy

Committed files contain manifests and summaries only. Raw corpora, processed corpora, tokenized arrays, tokenizer
model JSON, checkpoints, and generated runs stay ignored unless a later phase explicitly changes the policy.
