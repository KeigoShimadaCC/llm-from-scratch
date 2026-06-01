# 08. Inference And Sampling

## Goal

Understand how the trained checkpoint is loaded and sampled locally.

## Why It Matters

Inference turns logits into text. Sampling controls such as temperature, top-k, top-p, repetition penalty, stop
strings, and stop token ids can change output behavior without changing model weights.

## What This Part Does

Inference loads a checkpoint, encodes the prompt, runs the Transformer forward repeatedly, chooses the next token, and
decodes tokens back to text. Greedy decoding picks the strongest token. Temperature and top-k/top-p sampling allow
controlled randomness. Stop strings and stop token ids tell generation when to halt early.

## Repo Map

- [Inference runtime](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/inference/runtime.py): checkpoint
  loading, tokenizer loading, sampling, stopping, and result JSON.
- [Generate CLI](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/inference/generate.py): command-line
  wrapper for completions.
- [PHASE-11A inference config](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/configs/inference_corpus_v01.yaml):
  model/tokenizer/checkpoint paths for the current 30M smoke run.
- [Inference guide](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/phase08a_inference_guide.md):
  option matrix and CLI behavior.

## Run It

```bash
uv run python -m inference.generate --config configs/inference_corpus_v01.yaml --prompt "The model learns" --max-new-tokens 16 --seed 123
uv run python -m inference.generate --config configs/inference_corpus_v01.yaml --prompt "A local language model" --max-new-tokens 24 --temperature 0.8 --top-k 20 --seed 123
```

## Inspect It

The output JSON reports prompt token count, generated token count, latency, tokens/sec, decoding settings, checkpoint
path, and stop reason.

Example output shape:

```json
{
  "prompt": "The model learns",
  "generated_text": "The model learns.",
  "generated_tokens": 1,
  "tokens_per_sec": 100.0,
  "stop_reason": "stop_string_or_eos"
}
```

Exact timing depends on the machine. The important part is that the response records how generation was produced.

## Try Changing

Run the same prompt with greedy decoding (`--temperature 0`) and stochastic sampling (`--temperature 0.8 --top-k 20`).
Compare whether the output becomes more varied or less stable.

## Further Reading

- [Language model](https://en.wikipedia.org/wiki/Language_model)
- [Nucleus sampling](https://arxiv.org/abs/1904.09751)
