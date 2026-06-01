# 02. Data Sourcing And Licenses

## Goal

Understand how the project chooses public data sources and why raw, processed, and tokenized data stay out of git.

## Why It Matters

Language model behavior is limited by its corpus. A serious educational project needs data provenance, license notes,
attribution requirements, storage policy, and explicit exclusions before training.

## What This Part Does

This phase decides what data is allowed before any large download happens. It is a guardrail: the model can learn from
public text, but the repo should know where each source came from, what terms apply, what attribution is required, and
where local copies live. The committed registry is metadata-only; corpus payloads stay ignored under `data/**`.

## Repo Map

- [Corpus config](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/configs/corpus_v01.yaml): approved
  sources, paths, limits, license notes, and blocker policy.
- [Source manifest](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/corpus_v01_source_manifest.md):
  committed summary generated from the config.
- [Data card](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/DATA_CARD.md): plain-English statement
  of current data scope and limitations.
- [Data README](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/data/README.md): local raw/processed/tokenized
  storage convention.

## Run It

```bash
uv run python -m corpus.audit_sources --config configs/corpus_v01.yaml --output docs/corpus_v01_source_manifest.md
uv run python -m corpus.download --config configs/corpus_v01.yaml --dry-run
```

## Inspect It

The approved source categories are English Wikipedia, Japanese Wikipedia, Project Gutenberg, and Aozora Bunko.
Tatoeba is explicitly deferred for future evaluation or SFT helper data, not part of `corpus_v01` pretraining.

Example manifest result:

```text
Audit mode: metadata-only; no corpus payloads downloaded.
Approved source ids are exactly: aozora_bunko, enwiki, jawiki, project_gutenberg.
Raw root: data/raw/corpus_v01
Processed root: data/processed/corpus_v01
Tokenized root: data/tokenized/corpus_v01
```

That means the repo is ready to download data later, but has not committed raw corpus text.

## Try Changing

Add a hypothetical private or unverifiable source to a copy of the corpus config and run the source audit. The audit
should reject it unless provenance, license, attribution, and storage policy are documented.

## Further Reading

- [Wikipedia database downloads](https://dumps.wikimedia.org/)
- [Project Gutenberg](https://www.gutenberg.org/)
- [Aozora Bunko](https://www.aozora.gr.jp/)
