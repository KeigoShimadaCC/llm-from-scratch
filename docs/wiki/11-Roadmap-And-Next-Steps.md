# 11. Roadmap And Next Steps

## Goal

Understand what is complete and what would most improve the project next.

## Why It Matters

The educational lab is complete, but model quality is still limited by smoke-sized data and short training. The next
steps should improve data scale, training duration, SFT quality, evaluation depth, and Mac-native performance without
weakening the scratch-owned core.

## What This Part Does

This page prevents the course from ending with "done" when the model is still small. It separates the completed
learning lab from the next research-engineering tasks. The most important next improvement is real data scale plus
longer pretraining; better SFT and MLX parity matter after the base model has stronger language behavior.

## Repo Map

- [Final Write-up](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/FINAL_WRITEUP.md): current
  achieved-vs-deferred positioning.
- [PHASE-11A plan](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/phase-plans/PHASE-11A-MEANINGFUL-30M-TRAINING.md):
  latest meaningful-training phase spec.
- [MLX deferral](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/phase08a_mlx_deferral.md): why MLX is
  not yet claimed complete.
- [Current progress](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/PROGRESS.md): latest repo state and
  validation evidence.

## Run It

```bash
./bin/agentic status --repo-root .
uv run python -m eval.compare_checkpoints --manifest docs/checkpoint_manifest_corpus_v01.json --output docs/phase11a_real_corpus_checkpoint_comparison.md
```

## Inspect It

The runner should show no queued phases for the completed roadmap. The PHASE-11A report should live-evaluate the
local 30M checkpoint when ignored artifacts exist.

Practical priority order:

1. Full corpus download and cleaning with manifest summaries.
2. Tokenizer revisit after real corpus scale.
3. Longer `kgpt-30m` training with periodic comparison reports.
4. Better SFT dataset only after the base model improves.
5. Larger and more realistic eval set.
6. MLX model loading plus logits parity.

## Try Changing

Draft a future phase plan for one improvement: full corpus download, longer training, better SFT data, stronger eval,
MLX parity, or tokenizer revisit after real data scale.

## Further Reading

- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [Apple MLX GitHub](https://github.com/ml-explore/mlx)
- [nanoGPT reference project](https://github.com/karpathy/nanoGPT)
