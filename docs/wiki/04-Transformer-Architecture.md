# 04. Transformer Architecture

## Goal

Understand the scratch-owned decoder-only Transformer used by the project.

## Why It Matters

The architecture converts token ids into embeddings, applies causal self-attention and MLP blocks, then predicts the
next token. This is the core mechanism behind GPT-style language modeling.

## Repo Map

- [Transformer implementation](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/kgpt/transformer.py)
- [Architecture report](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/phase03a_transformer_architecture.md)
- [Micro Transformer config](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/configs/transformer_micro.yaml)
- [30M corpus config](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/configs/kgpt_30m_corpus_v01.yaml)

## Run It

```bash
uv run python -m train.transformer_smoke --config configs/transformer_micro.yaml --max-steps 20
uv run python -m inference.generate --config configs/transformer_micro.yaml --prompt hello --max-new-tokens 16 --seed 123
```

## Inspect It

Look for token embeddings, positional embeddings, causal masking, multi-head attention, MLP blocks, residual paths,
LayerNorm, and tied input/output embeddings in `kgpt/transformer.py`.

## Try Changing

Compare the parameter count of the micro config with `configs/kgpt_30m_corpus_v01.yaml`. Identify which fields drive
most of the scale increase: embedding dimension, number of layers, number of heads, and MLP hidden dimension.

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Transformer architecture](https://en.wikipedia.org/wiki/Transformer_%28deep_learning%29)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
