# Appendix. Glossary

## Goal

Define common terms used across the wiki.

## Why It Matters

LLM projects mix data, architecture, training, and systems terms. Shared definitions make the lessons easier to
follow.

## What This Part Does

Use this page when a lesson uses a term before it feels natural. Each definition is intentionally short. The better
learning move is to pair the definition with one source file or report where the term appears in practice.

## Repo Map

- [North Star](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/NORTH_STAR_LLM_FROM_SCRATCH_MAC.md):
  project vocabulary and scope.
- [Model Card](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/MODEL_CARD.md): model behavior and
  limitation terms.
- [Data Card](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/DATA_CARD.md): corpus and provenance
  terms.

## Terms

- Token: an integer id representing a text piece.
- BPE: byte pair encoding, a subword tokenization algorithm.
- Byte fallback: ability to represent unseen Unicode through UTF-8 bytes.
- Embedding: learned vector lookup for token ids.
- Logits: raw model scores before softmax.
- Causal mask: attention mask that prevents looking at future tokens.
- Attention head: one projection group inside multi-head self-attention.
- Loss: training objective value, here next-token cross entropy.
- Perplexity: exponentiated loss, useful as a language-model metric.
- Checkpoint: saved model and optimizer state plus metadata.
- Corpus: text collection used for tokenizer training or pretraining.
- Leakage: validation/test information appearing in training data.
- SFT: supervised fine-tuning, usually instruction/response training.
- MPS: Metal Performance Shaders backend used by PyTorch on Apple Silicon.
- MLX: Apple machine learning array framework for Apple Silicon.

## Run It

```bash
uv run python -m train.pretrain --config configs/kgpt_30m_corpus_v01.yaml --dry-run --validate-resume
```

Use the output to identify terms such as checkpoint metadata, tokenizer vocabulary, split validation, and parameter
count.

## Inspect It

Open the model card and data card, then match each claim to a command or artifact in the command and artifact indexes.

## Try Changing

Pick one glossary term and trace it to one concrete source file in the repo.

## Further Reading

- [Language model](https://en.wikipedia.org/wiki/Language_model)
- [Transformer architecture](https://en.wikipedia.org/wiki/Transformer_%28deep_learning%29)
