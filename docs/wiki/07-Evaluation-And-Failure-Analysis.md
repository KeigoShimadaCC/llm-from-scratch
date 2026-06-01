# 07. Evaluation And Failure Analysis

## Goal

Understand how the repo evaluates checkpoints without relying only on subjective samples.

## Why It Matters

Small LLMs can look interesting in one prompt and fail immediately in another. The repo uses fixed prompts, validation
loss, perplexity, exact-match toy tasks, repetition rate, sample snapshots, and failure labels to make limitations
visible.

## What This Part Does

Evaluation turns "it looks okay" into repeatable evidence. The same prompt set is sampled across checkpoints, metrics
are computed in one place, and obvious failure modes are named consistently. This makes model progress visible even
when the model is still weak.

## Repo Map

- [Evaluation config](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/configs/eval_fixed_prompts.yaml):
  fixed prompts, categories, and expected toy answers.
- [Checkpoint evaluator](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/eval/checkpoint_eval.py): loads
  checkpoints, samples prompts, computes metrics, and labels failures.
- [PHASE-07A eval report](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/phase07a_eval_report.md):
  single-checkpoint report.
- [Checkpoint comparison](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/phase07a_checkpoint_comparison.md):
  cross-checkpoint comparison for earlier phases.

## Run It

```bash
uv run python -m eval.report --config configs/eval_fixed_prompts.yaml --output docs/phase07a_eval_report.md
uv run python -m eval.compare_checkpoints --manifest docs/checkpoint_manifest.json --output docs/phase07a_checkpoint_comparison.md
```

## Inspect It

Look for failure classes such as `instruction_ignored`, `mode_collapse`, `pure_gibberish`, `language_mixing`, and
`repetition_loop`. These labels are not insults; they are diagnostics.

Example comparison result:

```text
Toy exact match: 0.00%
Failure classes: instruction_ignored, mode_collapse, pure_gibberish
```

That result is useful because it prevents a single nice-looking sample from hiding the model's actual limitations.

## Try Changing

Add one fixed prompt locally to `configs/eval_fixed_prompts.yaml`, rerun the eval report, and compare how each
checkpoint behaves. Do not commit generated checkpoint artifacts.

## Further Reading

- [Perplexity](https://en.wikipedia.org/wiki/Perplexity)
- [Evaluation of machine translation](https://en.wikipedia.org/wiki/Evaluation_of_machine_translation)
