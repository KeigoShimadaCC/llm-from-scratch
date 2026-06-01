# 06. The 30M Corpus Run

## Goal

Understand the current main model evidence: the PHASE-11A `kgpt-30m-corpus-v01` smoke run.

## Why It Matters

The 30M run proves the repo can scale beyond toy models while still using scratch-owned weights, configs, tokenizer
artifacts, training loop, evaluation, and inference. It is evidence of the lab pipeline, not a claim of useful chatbot
quality.

## What This Part Does

This is the current "main model" evidence row. It combines the corpus registry, cleaned smoke corpus, selected
byte-BPE tokenizer, 30M Transformer config, training loop, checkpoint format, fixed-prompt evaluation, and inference
CLI. The run is short enough to be educational and reproducible, but too small to produce strong language ability.

## Repo Map

- [30M corpus config](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/configs/kgpt_30m_corpus_v01.yaml):
  model size, optimizer settings, tokenized data paths, and run defaults.
- [Checkpoint manifest](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/checkpoint_manifest_corpus_v01.json):
  tells evaluation which checkpoints to compare.
- [PHASE-11A comparison report](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/phase11a_real_corpus_checkpoint_comparison.md):
  committed metrics, samples, and failure labels.
- [Inference config](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/configs/inference_corpus_v01.yaml):
  points generation at the PHASE-11A checkpoint and tokenizer.

## Run It

```bash
uv run python -m inference.generate --config configs/inference_corpus_v01.yaml --prompt "The model learns" --max-new-tokens 16 --seed 123
uv run python -m inference.generate --config configs/inference_corpus_v01.yaml --prompt "小さなモデル" --max-new-tokens 24 --seed 123
```

## Inspect It

PHASE-11A recorded:

- Parameters: 31,692,800
- Steps: 1000
- Tokens seen: 16,000
- Initial validation loss: 319.7539
- Final validation loss: 32.7539
- Best validation loss: 17.7304 at step 100
- Loss improvement: 89.76%
- Final CPU throughput: 344.53 tokens/sec during training

Example completions include:

- `The model learns` -> `The model learns.`
- `A local language model` -> `A local language models use attention to predict text. They`
- `小さなモデル` -> `小さなモデルめの短い文章です。`

Read these as pipeline evidence, not product quality. The validation loss improved sharply, but the fixed-prompt
report still labels failures such as `instruction_ignored`, `mode_collapse`, and `pure_gibberish`.

## Try Changing

Try a prompt that is outside the smoke corpus style, such as an instruction. The model will usually ignore the
instruction because PHASE-11A is pretraining evidence, not instruction-following quality evidence.

## Further Reading

- [Scaling law](https://en.wikipedia.org/wiki/Neural_scaling_law)
- [nanoGPT reference project](https://github.com/karpathy/nanoGPT)
