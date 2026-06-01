# 02. Data Sourcing And Licenses

## Goal

Understand how the project chooses public data sources and why raw, processed, and tokenized data stay out of git.

## Why It Matters

Language model behavior is limited by its corpus. A serious educational project needs data provenance, license notes,
attribution requirements, storage policy, and explicit exclusions before training.

## Repo Map

- [Corpus config](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/configs/corpus_v01.yaml)
- [Source manifest](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/corpus_v01_source_manifest.md)
- [Data card](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/DATA_CARD.md)
- [Data README](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/data/README.md)

## Run It

```bash
uv run python -m corpus.audit_sources --config configs/corpus_v01.yaml --output docs/corpus_v01_source_manifest.md
uv run python -m corpus.download --config configs/corpus_v01.yaml --dry-run
```

## Inspect It

The approved source categories are English Wikipedia, Japanese Wikipedia, Project Gutenberg, and Aozora Bunko.
Tatoeba is explicitly deferred for future evaluation or SFT helper data, not part of `corpus_v01` pretraining.

## Try Changing

Add a hypothetical private or unverifiable source to a copy of the corpus config and run the source audit. The audit
should reject it unless provenance, license, attribution, and storage policy are documented.

## Further Reading

- [Wikipedia database downloads](https://dumps.wikimedia.org/)
- [Project Gutenberg](https://www.gutenberg.org/)
- [Aozora Bunko](https://www.aozora.gr.jp/)
