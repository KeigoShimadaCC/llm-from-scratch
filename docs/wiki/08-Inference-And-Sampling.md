# 08. Inference And Sampling

## Goal

Understand how the trained checkpoint is loaded and sampled locally.

## Why It Matters

Inference turns logits into text. Sampling controls such as temperature, top-k, top-p, repetition penalty, stop
strings, and stop token ids can change output behavior without changing model weights.

## Repo Map

- [Inference runtime](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/inference/runtime.py)
- [Generate CLI](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/inference/generate.py)
- [PHASE-11A inference config](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/configs/inference_corpus_v01.yaml)
- [Inference guide](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/phase08a_inference_guide.md)

## Run It

```bash
uv run python -m inference.generate --config configs/inference_corpus_v01.yaml --prompt "The model learns" --max-new-tokens 16 --seed 123
uv run python -m inference.generate --config configs/inference_corpus_v01.yaml --prompt "A local language model" --max-new-tokens 24 --temperature 0.8 --top-k 20 --seed 123
```

## Inspect It

The output JSON reports prompt token count, generated token count, latency, tokens/sec, decoding settings, checkpoint
path, and stop reason.

## Try Changing

Run the same prompt with greedy decoding (`--temperature 0`) and stochastic sampling (`--temperature 0.8 --top-k 20`).
Compare whether the output becomes more varied or less stable.

## Further Reading

- [Language model](https://en.wikipedia.org/wiki/Language_model)
- [Nucleus sampling](https://arxiv.org/abs/1904.09751)
