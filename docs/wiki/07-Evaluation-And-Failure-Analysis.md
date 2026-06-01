# 07. Evaluation And Failure Analysis

## Goal

Understand how the repo evaluates checkpoints without relying only on subjective samples.

## Why It Matters

Small LLMs can look interesting in one prompt and fail immediately in another. The repo uses fixed prompts, validation
loss, perplexity, exact-match toy tasks, repetition rate, sample snapshots, and failure labels to make limitations
visible.

## Repo Map

- [Evaluation config](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/configs/eval_fixed_prompts.yaml)
- [Checkpoint evaluator](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/eval/checkpoint_eval.py)
- [PHASE-07A eval report](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/phase07a_eval_report.md)
- [Checkpoint comparison](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/phase07a_checkpoint_comparison.md)

## Run It

```bash
uv run python -m eval.report --config configs/eval_fixed_prompts.yaml --output docs/phase07a_eval_report.md
uv run python -m eval.compare_checkpoints --manifest docs/checkpoint_manifest.json --output docs/phase07a_checkpoint_comparison.md
```

## Inspect It

Look for failure classes such as `instruction_ignored`, `mode_collapse`, `pure_gibberish`, `language_mixing`, and
`repetition_loop`. These labels are not insults; they are diagnostics.

## Try Changing

Add one fixed prompt locally to `configs/eval_fixed_prompts.yaml`, rerun the eval report, and compare how each
checkpoint behaves. Do not commit generated checkpoint artifacts.

## Further Reading

- [Perplexity](https://en.wikipedia.org/wiki/Perplexity)
- [Evaluation of machine translation](https://en.wikipedia.org/wiki/Evaluation_of_machine_translation)
