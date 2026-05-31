# Data Directory

This directory documents data conventions for KeigoGPT-Lab.

Local data is intentionally ignored by git:

- `data/raw/`
- `data/processed/`
- `data/tokenized/`

Every dataset used for training should have a documented source, license status, preprocessing command, train/validation split method, and tokenizer compatibility note. Keep clean, inspectable data ahead of raw scale.

Do not commit private corpora, downloaded datasets, tokenized binary files, or generated data artifacts.

PHASE-00B uses no real corpus. `train.dummy` creates fake token batches in memory for smoke testing only.

PHASE-01A uses a repo-authored synthetic character fixture embedded in `configs/micro_char.yaml`:

```text
hello microgpt.
```

The line is repeated to create a tiny, low-entropy overfit corpus. Source/provenance: authored for this
repository, with no external text source or third-party license dependency. The fixture is intentionally
small and is only for tokenizer roundtrip tests, training-loop smoke tests, deterministic generation checks,
and the first overfitting curve.

## PHASE-02A Tokenizer And Tokenized Data Policy

PHASE-02A uses a repo-authored synthetic bilingual smoke corpus in `configs/tokenizer_bilingual.yaml`.
It is not a training claim corpus; it only exercises tokenizer quality reporting, split generation,
deduplication, leakage checks, token-file writing, and batch sampling.

Every data source manifest must include:

- `source_name`
- `url_or_local_note`
- `license`
- `language_mix`
- `size`
- `checksum`
- `preprocessing_command`
- `split_method`
- `dedup_strategy`
- `contamination_leakage_notes`

The PHASE-02A manifest is generated at `docs/phase02a_data_manifest.json` by:

```sh
uv run python -m train.sample_batches --config configs/tokenized_smoke.yaml --max-batches 2
```

Tokenized split files are generated under ignored `data/tokenized/phase02a_smoke/` and use:

- format: NumPy `.npy`
- layout: 1-D contiguous token-id stream per split
- dtype: `uint16` when the tokenizer vocabulary fits, otherwise `uint32`
- split names: `train` and `validation`
- sidecar: `data/tokenized/phase02a_smoke/metadata.json`

The metadata sidecar records tokenizer id, tokenizer model path, token file paths, dtype, token counts,
SHA-256 checksums, split record ids, exact normalized-text leakage check results, and the target semantics:
language-model targets are the same token stream shifted by one token.

Tokenizer model artifacts are generated under ignored `data/tokenized/tokenizers/`. The committed
tokenizer report lists the ignored model path and checksum so the artifact can be regenerated without
committing data-derived files.
