# 10. Hands-On Labs

## Goal

Use short exercises to connect concepts to actual commands.

## Why It Matters

This project is best learned by running it. The point is to inspect the artifacts, not only read explanations.

## What This Part Does

The labs are short loops: run one command, inspect one artifact, make one small change, and explain what changed.
They are intentionally smoke-sized so a student can focus on the mechanics before attempting larger training.

## Repo Map

- [Command Index](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/COMMAND_INDEX.md): canonical
  command list.
- [Artifact Index](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/ARTIFACT_INDEX.md): where evidence
  lives and how to recreate it.
- [PHASE-11A report](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/phase11a_real_corpus_checkpoint_comparison.md):
  current 30M smoke-run metrics, samples, and failure labels.

## Run It

### Lab 1: Run Inference

```bash
uv run python -m inference.generate --config configs/inference_corpus_v01.yaml --prompt "The model learns" --max-new-tokens 16 --seed 123
```

Expected shape: JSON with `generated_text`, token counts, `latency_sec`, and `tokens_per_sec`.

### Lab 2: Change Sampling

```bash
uv run python -m inference.generate --config configs/inference_corpus_v01.yaml --prompt "A local language model" --max-new-tokens 24 --temperature 0.8 --top-k 20 --seed 123
```

Expected shape: same JSON fields, but completion text may differ.

### Lab 3: Regenerate Tokenizer Report

```bash
uv run python -m tokenizer.train_report --config configs/tokenizer_corpus_v01.yaml --output docs/tokenizer_corpus_v01_report.md
```

Expected shape: Markdown report with vocabulary sweep, English/Japanese stats, and byte fallback notes.

### Lab 4: Sample Token Batches

```bash
uv run python -m train.sample_batches --config configs/corpus_v01_tokenized.yaml --max-batches 2
```

Expected shape: JSON-like batch summaries with shapes, token ids, and metadata.

### Lab 5: Validate The 30M Config

```bash
uv run python -m train.pretrain --config configs/kgpt_30m_corpus_v01.yaml --dry-run --validate-resume
```

Expected shape: JSON with parameter count, tokenizer compatibility, split checks, and resume checkpoint metadata.

### Lab 6: Inspect Checkpoint Comparison

```bash
uv run python -m eval.compare_checkpoints --manifest docs/checkpoint_manifest_corpus_v01.json --output docs/phase11a_real_corpus_checkpoint_comparison.md
```

Expected shape: Markdown table with checkpoint status, parameter count, validation loss, perplexity, tokens/sec, toy
exact-match rate, sample snapshots, and failure classes. If ignored checkpoint artifacts are absent in a fresh clone,
the report may use committed summary data instead of live evaluation.

## Inspect It

After each lab, open the referenced report or JSON payload and ask: what artifact proves the command worked?

Example evidence to notice:

- Inference: `generated_text`, generated token count, and stop reason.
- Tokenizer report: selected tokenizer, actual vocabulary size, English/Japanese token counts.
- 30M dry-run: `31,692,800` parameters and resume validation.
- Checkpoint comparison: failure labels such as `instruction_ignored` and `mode_collapse`.

## Try Changing

Run the same lab twice with different seeds. For greedy decoding, seed may not affect output. For stochastic sampling,
seed should matter.

## Further Reading

- [Command Index](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/COMMAND_INDEX.md)
- [Artifact Index](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/ARTIFACT_INDEX.md)
