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

1. Start with [[00. What From Scratch Means|00-What-From-Scratch-Means]].
2. Follow the data path through tokenization, source registry, cleaning, splitting, and leakage checks.
3. Study the Transformer architecture and training loop.
4. Inspect the 30M corpus smoke run and evaluation reports.
5. Run local inference and try the labs.
6. Review the roadmap for full-corpus training, better SFT, and MLX parity.

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
