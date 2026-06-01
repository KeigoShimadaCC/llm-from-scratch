# 01. Text To Tokens

## Goal

Understand why an LLM does not read raw strings directly and how this repo uses a byte-BPE tokenizer for English and
Japanese smoke data.

## Why It Matters

The model learns over integer token ids. Tokenization controls vocabulary size, context efficiency, byte fallback,
and how badly Japanese or mixed text fragments. Poor tokenization can make training less efficient even when the
Transformer implementation is correct.

## Repo Map

- [Tokenizer config](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/configs/tokenizer_corpus_v01.yaml)
- [Tokenizer report](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/tokenizer_corpus_v01_report.md)
- [Byte-BPE implementation](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/kgpt/byte_bpe.py)
- [Token data pipeline](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/kgpt/token_data.py)

## Run It

```bash
uv run python -m tokenizer.train_report --config configs/tokenizer_corpus_v01.yaml --output docs/tokenizer_corpus_v01_report.md
uv run python -m train.sample_batches --config configs/corpus_v01_tokenized.yaml --max-batches 2
```

## Inspect It

Open the tokenizer report and compare English, Japanese, and mixed text token counts. The PHASE-10D smoke tokenizer
is named `kgpt-corpus-v01-byte-bpe-4k`, but the tiny smoke corpus caps the actual vocabulary at 312 tokens.

## Try Changing

Change the prompt passed to `inference.generate` from English to Japanese and inspect the output. Then compare that
behavior with the tokenizer report's Japanese segmentation examples.

## Further Reading

- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [Byte pair encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding)
