# PHASE-02A Tokenizer Report

## Summary

- Selected tokenizer: `phase02a-byte-bpe-bilingual`.
- Algorithm: repo-owned byte-level BPE with full UTF-8 byte fallback.
- Vocabulary size: 377 tokens (117 learned merges plus 256 byte tokens and special tokens).
- Config: `configs/tokenizer_bilingual.yaml`.
- Report output: `docs/tokenizer_report.md`.
- Ignored tokenizer model artifact: `data/tokenized/tokenizers/phase02a-byte-bpe-bilingual.json`.
- Tokenizer model sha256: `29a57bf7b77419471233a2045059b44f509a968094986cbea03546762166e14d`.

The final tokenizer choice for larger training is deferred. This phase only proves the token-level pipeline on a repo-authored bilingual smoke corpus; larger approved data is required before PHASE-04A/PHASE-05A should treat this tokenizer as final.

## Candidate Status

| Candidate | Status | Rationale |
| --- | --- | --- |
| `byte_bpe` | implemented | Repo-owned UTF-8 byte-level BPE gives deterministic roundtrip behavior without adding phase-forbidden dependencies. |
| `sentencepiece_unigram` | rejected-for-phase | Requires adding the sentencepiece dependency and generated model artifacts; pyproject.toml and uv.lock are outside PHASE-02A allowed paths. |
| `sentencepiece_bpe` | rejected-for-phase | Same dependency and lockfile constraint as SentencePiece unigram; revisit after dependency changes are explicitly in scope. |

## Vocabulary Sweep

Minimum pair frequency: 2. The North Star sizes 8k, 16k, and 32k are requested,
but this smoke corpus is intentionally small, so byte-level BPE stops when no more eligible pairs remain.

| Corpus | Requested vocab | Actual vocab | Learned merges | Status |
| --- | ---: | ---: | ---: | --- |
| english_only | 8000 | 323 | 63 | capped; no more pairs met min frequency 2 |
| english_only | 16000 | 323 | 63 | capped; no more pairs met min frequency 2 |
| english_only | 32000 | 323 | 63 | capped; no more pairs met min frequency 2 |
| bilingual | 8000 | 377 | 117 | capped; no more pairs met min frequency 2 |
| bilingual | 16000 | 377 | 117 | capped; no more pairs met min frequency 2 |
| bilingual | 32000 | 377 | 117 | capped; no more pairs met min frequency 2 |

## English/Japanese Tokenization

| Group | Sentences | Mean tokens/sentence | Mean bytes/token | Mean unknown tokens |
| --- | ---: | ---: | ---: | ---: |
| english | 2 | 26.50 | 1.62 | 0.00 |
| japanese | 2 | 34.00 | 1.83 | 0.00 |
| mixed | 2 | 30.00 | 1.88 | 0.00 |

## English-Only vs Bilingual Comparison

| Tokenizer corpus | Sentence group | Mean tokens/sentence | Mean bytes/token |
| --- | --- | ---: | ---: |
| english_only | english | 31.50 | 1.37 |
| english_only | japanese | 58.50 | 1.00 |
| english_only | mixed | 51.50 | 1.10 |
| bilingual | english | 26.50 | 1.62 |
| bilingual | japanese | 34.00 | 1.83 |
| bilingual | mixed | 30.00 | 1.88 |

## Unknown And Byte Fallback Behavior

The tokenizer reserves `<unk>`, but normal text encoding uses UTF-8 bytes, so unseen Unicode still roundtrips without producing unknown tokens.

| Probe | Tokens | Unknowns | Roundtrip |
| --- | ---: | ---: | --- |
| Unseen emoji 😀 and rare kanji 𠮷 still roundtrip. | 47 | 0 | pass |
| 未知の記号🧪もバイトで保持します。 | 32 | 0 | pass |

## Compression And Context-Length Effect

Bytes/token is a practical compression proxy: higher values mean each context window carries more source text.

| Sentence group | Context tokens | Approx UTF-8 bytes/context | Approx characters/context |
| --- | ---: | ---: | ---: |
| english | 128 | 207.3 | 207.3 |
| english | 512 | 829.1 | 829.1 |
| japanese | 128 | 234.5 | 78.2 |
| japanese | 512 | 938.2 | 312.7 |
| mixed | 128 | 241.0 | 160.2 |
| mixed | 512 | 963.9 | 641.0 |

## Failure Cases And Segmentation Examples

These examples are expected rough edges for a tiny smoke corpus. Japanese text and emoji-heavy mixed text still fall back to byte fragments when the corpus has not repeated the same byte patterns enough to merge them.

- `形態素解析なしの日本語固有名詞「山田太郎」は細かく割れやすい。`
  - tokens: 68; bytes/token: 1.37; pieces: 0xe5, 0xbd, 0xa2, 0xe6, 0x85, 0x8b, 0xe7, 0xb4, 0xa0, 0xe8, 0xa7, 0xa3, 0xe6, 0x9e, 0x90, 0xe381, 0xaa, 0xe38197e381, 0xaee6, 0x97a5e69cace8aa9e, 0xe5, 0x9b, 0xba, 0xe69c, 0x89, 0xe5, 0x90, 0x8d, 0xe8, 0xa9, 0x9e, 0xe380, ...
- `Emoji 😀🧪 and mixed-script identifiers kgpt_日本語_v1 fragment under tiny data.`
  - tokens: 71; bytes/token: 1.23; pieces: E, m, o, j, i, ' ', 0xf0, 0x9f, 0x98, 0x80, 0xf0, 0x9f, 0xa7, 0xaa, ' ', an, d, ' ', m, i, x, e, d, -, s, c, r, i, p, t, ' ', i, ...

## Sentence Samples

| Group | Sentence | Tokens | Bytes/token |
| --- | --- | ---: | ---: |
| english | The model learns from clean local data. | 25 | 1.56 |
| english | Tokenization changes the useful context length. | 28 | 1.68 |
| japanese | 小さなモデルが日本語と英語を学習します。 | 26 | 2.31 |
| japanese | 文脈長が短いと分割の粗さが目立ちます。 | 42 | 1.36 |
| mixed | KeigoGPT は local Mac で training を試します。 | 29 | 1.86 |
| mixed | tokenizer report は English と日本語を比べます。 | 31 | 1.90 |
