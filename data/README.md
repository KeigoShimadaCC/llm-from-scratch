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

## PHASE-10A/10B corpus_v01 policy

`corpus_v01` is limited to four audited public source categories:

- English Wikipedia
- Japanese Wikipedia
- Project Gutenberg
- Aozora Bunko

PHASE-10A commits only source registry metadata and the generated source manifest. PHASE-10B adds
CI-safe download planning and smoke cleaning:

```sh
uv run python -m corpus.download --config configs/corpus_v01.yaml --dry-run
uv run python -m corpus.clean --config configs/corpus_v01.yaml --smoke --output data/processed/corpus_v01_smoke
```

`corpus.download --dry-run` does not fetch payloads. It validates the audited source registry and prints
the local raw/processed paths, source locators, checksum policy, and eligibility status that a supervised
download would use later.

`corpus.clean --smoke` writes a repo-authored smoke corpus under ignored
`data/processed/corpus_v01_smoke/`:

- `documents.jsonl`: processed records with `doc_id`, `source_id`, `lang`, `title`, `text`, `license`,
  `attribution`, `source_url`, `source_record_id`, `sha256`, and `cleaning_version`
- `manifest.json`: aggregate source/language counts and the JSONL checksum

The smoke corpus exercises Wikipedia, Gutenberg, and Aozora cleaning behavior without committing upstream
corpus text. Full raw downloads, full processed corpora, tokenized arrays, tokenizer artifacts, and
checkpoints remain ignored local artifacts.

## PHASE-10C split and leakage manifest

PHASE-10C consumes the ignored PHASE-10B smoke processed corpus and writes a committed manifest summary:

```sh
uv run python -m corpus.split_manifest --config configs/corpus_v01.yaml --processed data/processed/corpus_v01_smoke --output docs/corpus_v01_dataset_manifest.json
```

The split is document-level and deterministic. Records are exact-deduplicated by normalized text hash
before assignment to `train`, `validation`, and `test`. Leakage checks fail if a normalized text hash or
source record id appears across splits.

The committed manifest stores aggregate counts, hashes, split membership, byte/character totals, source
counts, language counts, dedup counts, and leakage results. It must not contain full document text.
