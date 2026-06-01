# 00. What From Scratch Means

## Goal

Understand the boundary of this project: what the repo builds itself, what it delegates to PyTorch, and what it
explicitly does not claim.

## Why It Matters

"From scratch" can be misleading. In this project it means the model architecture, tokenizer path, data pipeline,
training loop, evaluation harness, inference path, and weights are owned by the repo. It does not mean hand-writing
matrix multiplication kernels or pretending a local Mac can reproduce frontier-scale pretraining.

## What This Part Does

This lesson sets the boundary for the rest of the course. Later lessons are allowed to use PyTorch tensors, YAML
configs, pytest, and GitHub Actions, but the core path should still be inspectable: repo-owned tokenizer, repo-owned
model code, repo-owned training loop, repo-trained weights, and repo-owned inference.

## Repo Map

- [README](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/README.md): current status and navigation.
- [North Star](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/NORTH_STAR_LLM_FROM_SCRATCH_MAC.md):
  project definition, non-goals, and success criteria.
- [Final Write-up](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/FINAL_WRITEUP.md): claim-to-evidence
  summary after the first end-to-end pass.
- [Model Card](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/MODEL_CARD.md): honest model capability
  and limitation summary.

## Run It

```bash
uv run pytest
uv run ruff check .
```

These commands verify that the project-owned implementation and tests still pass.

## Inspect It

Expected result shape:

```text
<number> passed
All checks passed!
```

Open the North Star and final write-up. Look for the distinction between `scratch-core`, reference baselines, and
practical local LLM experiments. That distinction is why this repo can be educational without pretending to be a
production chatbot.

## Try Changing

Write down one feature that would violate the scratch boundary, such as replacing the core model with a Hugging Face
`AutoModelForCausalLM`. Then find where the repo documents that this is out of scope.

## Further Reading

- [Generative pre-trained transformer](https://en.wikipedia.org/wiki/Generative_pre-trained_transformer)
- [nanoGPT reference project](https://github.com/karpathy/nanoGPT)
