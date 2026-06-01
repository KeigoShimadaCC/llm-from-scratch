# PHASE-10D corpus_v01 Tokenizer Report

## Summary

- Selected tokenizer: `kgpt-corpus-v01-byte-bpe-4k`.
- Algorithm: repo-owned byte-level BPE with full UTF-8 byte fallback.
- Vocabulary size: 312 tokens (52 learned merges plus 256 byte tokens and special tokens).
- Config: `configs/tokenizer_corpus_v01.yaml`.
- Report output: `docs/tokenizer_corpus_v01_report.md`.
- Ignored tokenizer model artifact: `data/tokenized/corpus_v01_tokenizers/byte_bpe_4k.json`.
- Tokenizer model sha256: `f746d83e005915c2e9d94f8b1f463cb7697e93656b55d49f8533e5212ad20e1d`.

PHASE-10D selects a conservative byte-BPE tokenizer for corpus_v01 smoke evidence. The chosen tokenizer is sufficient to unblock tokenized data and PHASE-11A smoke training, but full-corpus tokenization should be revisited when real local downloads are available.


## Candidate Status

| Candidate | Status | Rationale |
| --- | --- | --- |
| `byte_bpe_2k` | compared | Smallest embedding budget; useful if 30M parameter budget is tight. |
| `byte_bpe_4k` | selected-for-corpus-v01-smoke | Best conservative tradeoff for the smoke corpus and 30M model budget. |
| `byte_bpe_8k` | compared | Candidate for better multilingual compression when the full corpus has enough repetition. |
| `byte_bpe_16k` | compared | Upper PHASE-10D candidate; likely too large for smoke evidence but useful as a gate. |

## Vocabulary Sweep

Minimum pair frequency: 2. Requested vocabulary sizes come from the phase config,
but smoke corpora are intentionally small, so byte-level BPE stops when no more eligible pairs remain.

| Corpus | Requested vocab | Actual vocab | Learned merges | Status |
| --- | ---: | ---: | ---: | --- |
| english_only | 2000 | 286 | 26 | capped; no more pairs met min frequency 2 |
| english_only | 4000 | 286 | 26 | capped; no more pairs met min frequency 2 |
| english_only | 8000 | 286 | 26 | capped; no more pairs met min frequency 2 |
| english_only | 16000 | 286 | 26 | capped; no more pairs met min frequency 2 |
| bilingual | 2000 | 312 | 52 | capped; no more pairs met min frequency 2 |
| bilingual | 4000 | 312 | 52 | capped; no more pairs met min frequency 2 |
| bilingual | 8000 | 312 | 52 | capped; no more pairs met min frequency 2 |
| bilingual | 16000 | 312 | 52 | capped; no more pairs met min frequency 2 |

## English/Japanese Tokenization

| Group | Sentences | Mean tokens/sentence | Mean bytes/token | Mean unknown tokens |
| --- | ---: | ---: | ---: | ---: |
| english | 2 | 53.00 | 1.20 | 0.00 |
| japanese | 2 | 43.50 | 1.43 | 0.00 |
| mixed | 2 | 49.00 | 1.25 | 0.00 |

## English-Only vs Bilingual Comparison

| Tokenizer corpus | Sentence group | Mean tokens/sentence | Mean bytes/token |
| --- | --- | ---: | ---: |
| english_only | english | 53.00 | 1.20 |
| english_only | japanese | 61.50 | 1.00 |
| english_only | mixed | 56.50 | 1.10 |
| bilingual | english | 53.00 | 1.20 |
| bilingual | japanese | 43.50 | 1.43 |
| bilingual | mixed | 49.00 | 1.25 |

## Unknown And Byte Fallback Behavior

The tokenizer reserves `<unk>`, but normal text encoding uses UTF-8 bytes, so unseen Unicode still roundtrips without producing unknown tokens.

| Probe | Tokens | Unknowns | Roundtrip |
| --- | ---: | ---: | --- |
| rare emoji 🧪🧠 | 16 | 0 | pass |
| 未知語とEnglishMix | 18 | 0 | pass |

## Compression And Context-Length Effect

Bytes/token is a practical compression proxy: higher values mean each context window carries more source text.

| Sentence group | Context tokens | Approx UTF-8 bytes/context | Approx characters/context |
| --- | ---: | ---: | ---: |
| english | 128 | 153.1 | 153.1 |
| english | 256 | 306.2 | 306.2 |
| english | 512 | 612.3 | 612.3 |
| japanese | 128 | 182.5 | 60.8 |
| japanese | 256 | 364.9 | 121.6 |
| japanese | 512 | 729.8 | 243.3 |
| mixed | 128 | 160.4 | 129.7 |
| mixed | 256 | 320.9 | 259.4 |
| mixed | 512 | 641.8 | 518.7 |

## Failure Cases And Segmentation Examples

These examples are expected rough edges for a tiny smoke corpus. Japanese text and emoji-heavy mixed text still fall back to byte fragments when the corpus has not repeated the same byte patterns enough to merge them.

- `超電導量子ビットとattention`
  - tokens: 31; bytes/token: 1.16; pieces: 0xe8, 0xb6, 0x85, 0xe9, 0x9b, 0xbb, 0xe5, 0xb0, 0x8e, 0xe9, 0x87, 0x8f, 0xe5, 0xad, 0x90, 0xe383, 0x93, 0xe383, 0x83, 0xe383, 0x88, 0xe381, 0xa8, a, t, t, en, t, i, o, n
- `Project Gutenberg boilerplate should be removed before tokenization.`
  - tokens: 56; bytes/token: 1.21; pieces: P, r, o, j, e, c, t, ' ', G, u, t, en, b, e, r, g, ' ', b, o, i, l, e, r, p, l, a, t, e , s, h, o, u, ...

## Sentence Samples

| Group | Sentence | Tokens | Bytes/token |
| --- | --- | ---: | ---: |
| english | The history of artificial intelligence began with symbolic reasoning. | 55 | 1.25 |
| english | A local language model learns from clean, documented data. | 51 | 1.14 |
| japanese | 人工知能の歴史は研究と実験の積み重ねです。 | 48 | 1.31 |
| japanese | 小さなモデルでもデータの品質は重要です。 | 39 | 1.54 |
| mixed | English and 日本語 can share one byte-level tokenizer. | 47 | 1.21 |
| mixed | KeigoGPT は local Mac で学習する educational model です。 | 51 | 1.29 |
