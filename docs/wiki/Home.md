# KeigoGPT-Lab: LLM From Scratch Course

This wiki is a hands-on course layer for
[KeigoGPT-Lab](https://github.com/KeigoShimadaCC/llm-from-scratch), a Mac-local LLM-from-scratch project.
It teaches the pipeline from text data to tokenizer, Transformer architecture, training, evaluation, and local
inference.

The current project is educationally complete as an end-to-end lab: the repo owns the tokenizer path, data pipeline,
decoder-only Transformer, training loop, evaluation harness, checkpoints, and inference CLI. It is not a production
chatbot, and it is not fully optimized: full-scale corpus training and MLX parity are still future work.

## Run The Current Model

From the repository root:

```bash
uv run python -m inference.generate --config configs/inference_corpus_v01.yaml --prompt "The model learns" --max-new-tokens 16 --seed 123
```

Expected shape: a JSON payload with `generated_text`, checkpoint path, token counts, latency, and throughput.

## Learning Path

Use each lesson in the same order: read the idea, inspect the repo files, run one command, then look at the artifact
that proves what happened. The goal is to connect a concept to a concrete file and a concrete result.

## Course Map

- [[00. What From Scratch Means|00-What-From-Scratch-Means]]: the project boundary and what this repo owns.
- [[01. Text To Tokens|01-Text-To-Tokens]]: how text becomes integer token ids.
- [[02. Data Sourcing And Licenses|02-Data-Sourcing-And-Licenses]]: where the corpus can come from and why provenance
  matters.
- [[03. Cleaning, Splitting, And Leakage|03-Cleaning-Splitting-And-Leakage]]: how raw text becomes train/validation/test
  records.
- [[04. Transformer Architecture|04-Transformer-Architecture]]: how embeddings, attention, MLPs, and logits fit
  together.
- [[05. Training Loop And Checkpoints|05-Training-Loop-And-Checkpoints]]: how next-token prediction updates weights
  and saves evidence.
- [[06. The 30M Corpus Run|06-The-30M-Corpus-Run]]: the current main educational model run and its limitations.
- [[07. Evaluation And Failure Analysis|07-Evaluation-And-Failure-Analysis]]: fixed prompts, metrics, and failure labels.
- [[08. Inference And Sampling|08-Inference-And-Sampling]]: how a checkpoint turns a prompt into generated text.
- [[09. Mac Optimization And MLX|09-Mac-Optimization-And-MLX]]: CPU/MPS benchmarking, KV-cache parity, and MLX deferral.
- [[10. Hands-On Labs|10-Hands-On-Labs]]: short exercises with expected output shapes.
- [[11. Roadmap And Next Steps|11-Roadmap-And-Next-Steps]]: what would make the model meaningfully better.
- [[Appendix. Command Index|Appendix-Command-Index]]: grouped commands.
- [[Appendix. Glossary|Appendix-Glossary]]: short definitions.

## How To Read A Lesson

- **Concept:** what the phase does in an LLM pipeline.
- **Repo files:** which files implement or document that part.
- **Command:** one small command that exercises the concept.
- **Evidence:** the report, JSON, checkpoint metadata, or sample output to inspect.
- **Experiment:** one change to try locally without committing generated artifacts.

## Repo Map

- [README](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/README.md)
- [North Star](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/NORTH_STAR_LLM_FROM_SCRATCH_MAC.md)
- [Command Index](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/COMMAND_INDEX.md)
- [Artifact Index](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/ARTIFACT_INDEX.md)
- [Final Write-up](https://github.com/KeigoShimadaCC/llm-from-scratch/blob/main/docs/FINAL_WRITEUP.md)

## Further Reading

- [Transformer architecture](https://en.wikipedia.org/wiki/Transformer_%28deep_learning%29)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [nanoGPT reference project](https://github.com/karpathy/nanoGPT)
