# 05. Training Loop And Checkpoints

## Goal

Understand how the repo trains from random initialization and records reproducible evidence.

## Why It Matters

Architecture alone is not an LLM. The training loop samples token batches, shifts targets by one token, computes
cross-entropy loss, updates weights, evaluates validation batches, records samples, and saves checkpoints.

## What This Part Does

Training is the part that changes random weights into a language model. For each batch, the model sees tokens
`x[0:n]` and learns to predict `x[1:n+1]`. The optimizer updates weights from the loss gradient. Periodic validation
checks whether the model is improving on held-out data, and checkpoint files make the run resumable and inspectable.

## Repo Map

- [Pretraining CLI](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/train/pretrain.py): training entry
  point, logging, validation, samples, and checkpoint writes.
- [Pretrain config parser](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/kgpt/pretrain.py): config
  schema and run setup helpers.
- [Tiny config](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/configs/kgpt_tiny.yaml): small early
  training config.
- [30M corpus config](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/configs/kgpt_30m_corpus_v01.yaml):
  current main model config.

## Run It

```bash
uv run python -m train.pretrain --config configs/kgpt_30m_corpus_v01.yaml --dry-run --validate-resume
uv run python -m train.pretrain --config configs/kgpt_30m_corpus_v01.yaml --max-steps 1000 --run-name phase11a_kgpt30m_corpus_v01_smoke
```

## Inspect It

The ignored run directory contains `config.yaml`, `metrics.jsonl`, `samples.txt`, `checkpoint_last.pt`,
`checkpoint_best.pt`, `tokenizer_info.json`, and `manifest.json`. These files are local evidence and must not be
committed.

Example dry-run result shape:

```json
{
  "parameter_count": 31692800,
  "tokenizer_compatible": true,
  "resume_check": "passed",
  "will_train": false
}
```

Example training evidence after a real run is `metrics.jsonl`: one JSON row per logged step with train loss,
validation loss, learning rate, tokens/sec, and tokens seen.

## Try Changing

Run the dry-run command first and inspect the JSON. It should report parameter count, tokenizer compatibility, split
token counts, leakage status, and resume checkpoint metadata without training.

## Further Reading

- [Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy)
- [Adam optimizer](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam)
