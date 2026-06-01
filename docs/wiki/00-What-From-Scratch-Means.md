# 00. What From Scratch Means

## Goal

Understand the boundary of this project: what the repo builds itself, what it delegates to PyTorch, and what it
explicitly does not claim.

## Why It Matters

"From scratch" can be misleading. In this project it means the model architecture, tokenizer path, data pipeline,
training loop, evaluation harness, inference path, and weights are owned by the repo. It does not mean hand-writing
matrix multiplication kernels or pretending a local Mac can reproduce frontier-scale pretraining.

## Repo Map

- [README](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/README.md)
- [North Star](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/NORTH_STAR_LLM_FROM_SCRATCH_MAC.md)
- [Final Write-up](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/FINAL_WRITEUP.md)
- [Model Card](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/MODEL_CARD.md)

## Run It

```bash
uv run pytest
uv run ruff check .
```

These commands verify that the project-owned implementation and tests still pass.

## Inspect It

Open the North Star and final write-up. Look for the distinction between `scratch-core`, reference baselines, and
practical local LLM experiments.

## Try Changing

Write down one feature that would violate the scratch boundary, such as replacing the core model with a Hugging Face
`AutoModelForCausalLM`. Then find where the repo documents that this is out of scope.

## Further Reading

- [Generative pre-trained transformer](https://en.wikipedia.org/wiki/Generative_pre-trained_transformer)
- [nanoGPT reference project](https://github.com/karpathy/nanoGPT)
