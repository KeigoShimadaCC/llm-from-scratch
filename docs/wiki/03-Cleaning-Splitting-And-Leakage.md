# 03. Cleaning, Splitting, And Leakage

## Goal

Understand the data pipeline after source selection: cleaning, normalization, deduplication, deterministic splitting,
and leakage checks.

## Why It Matters

Training metrics are only meaningful if validation text is held out and not duplicated from training text. Cleaning
also prevents markup, boilerplate, and malformed text from dominating a small educational run.

## Repo Map

- [Cleaning CLI](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/corpus/clean.py)
- [Cleaning helpers](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/corpus/cleaning.py)
- [Split manifest CLI](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/corpus/split_manifest.py)
- [Dataset manifest](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/corpus_v01_dataset_manifest.json)

## Run It

```bash
uv run python -m corpus.clean --config configs/corpus_v01.yaml --smoke --output data/processed/corpus_v01_smoke
uv run python -m corpus.split_manifest --config configs/corpus_v01.yaml --processed data/processed/corpus_v01_smoke --output docs/corpus_v01_dataset_manifest.json
```

## Inspect It

The committed dataset manifest stores summary metadata, hashes, split membership, and leakage counts. It does not
commit full corpus text.

## Try Changing

Duplicate one smoke record locally, rerun the split manifest command, and inspect the duplicate-removal summary.
Keep the generated processed files ignored.

## Further Reading

- [Data leakage](https://en.wikipedia.org/wiki/Leakage_%28machine_learning%29)
- [Unicode normalization](https://en.wikipedia.org/wiki/Unicode_equivalence)
