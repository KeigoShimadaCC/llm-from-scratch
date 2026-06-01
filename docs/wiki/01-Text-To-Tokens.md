# 01. Text To Tokens

## Goal

Understand why an LLM does not read raw strings directly and how this repo uses a byte-BPE tokenizer for English and
Japanese smoke data.

## Why It Matters

The model learns over integer token ids. Tokenization controls vocabulary size, context efficiency, byte fallback,
and how badly Japanese or mixed text fragments. Poor tokenization can make training less efficient even when the
Transformer implementation is correct.

## What This Part Does

The tokenizer converts text into token ids before training. This repo uses byte-level BPE for the main path: start
from UTF-8 bytes, repeatedly merge frequent adjacent byte sequences, then map the resulting pieces to integers. Byte
fallback matters because unseen Japanese text, emoji, or mixed text can still roundtrip instead of becoming unknown
tokens.

## Repo Map

- [Tokenizer config](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/configs/tokenizer_corpus_v01.yaml):
  candidate vocab sizes and report settings.
- [Tokenizer report](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/tokenizer_corpus_v01_report.md):
  selected tokenizer, English/Japanese stats, fallback behavior, and rough edges.
- [Byte-BPE implementation](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/kgpt/byte_bpe.py): training,
  encode, decode, merge table, and byte fallback.
- [Token data pipeline](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/kgpt/token_data.py): writes and
  reads token arrays for training batches.

## Run It

```bash
uv run python -m tokenizer.train_report --config configs/tokenizer_corpus_v01.yaml --output docs/tokenizer_corpus_v01_report.md
uv run python -m train.sample_batches --config configs/corpus_v01_tokenized.yaml --max-batches 2
```

## Inspect It

Open the tokenizer report and compare English, Japanese, and mixed text token counts. The PHASE-10D smoke tokenizer
is named `kgpt-corpus-v01-byte-bpe-4k`, but the tiny smoke corpus caps the actual vocabulary at 312 tokens.

Example report results:

| Probe | What To Notice |
| --- | --- |
| English sentence | `A local language model learns from clean, documented data.` becomes 51 tokens. |
| Japanese sentence | `小さなモデルでもデータの品質は重要です。` becomes 39 tokens. |
| Rare text | `rare emoji 🧪🧠` roundtrips with 0 unknown tokens because byte fallback handles unseen bytes. |

Byte-BPE pieces are not always whole words. In the smoke tokenizer, repeated English fragments can merge into pieces
like `en`, while rare Japanese fragments may still appear as UTF-8 byte pieces such as `0xe8`, `0xb6`, `0x85`.
That is not a bug in byte fallback; it is evidence that the smoke corpus is small.

## Try Changing

Change the prompt passed to `inference.generate` from English to Japanese and inspect the output. Then compare that
behavior with the tokenizer report's Japanese segmentation examples.

## Further Reading

- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [Byte pair encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding)
