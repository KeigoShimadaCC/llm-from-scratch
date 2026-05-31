# North Star And Vision

## Product Intent

`KeigoGPT-Lab` is a Mac-local, from-scratch decoder-only LLM laboratory. The project is educational, but technically serious: the repo should make the full path from text data to trained weights to local inference inspectable.

## North Star

Build a clean, reproducible, Apple Silicon-aware small LLM stack:

- tokenizer and data pipeline owned by this repo;
- decoder-only Transformer implemented in the scratch-core path;
- training loop, checkpointing, metrics, sampling, and evaluation owned by this repo;
- at least one meaningful small model trained from random initialization;
- instruction-tuning, failure analysis, and Mac-native inference explored after the core model works.

## Success Criteria

- A working decoder-only Transformer trains from random initialization.
- Character-level and token-level training paths are both understandable and tested.
- English/Japanese tokenizer choices are measured, not guessed.
- Experiments produce configs, metrics, checkpoints, generated samples, and short reports.
- Evaluation explains failure modes, not only loss curves.
- Inference runs locally on the Mac without cloud services.

## Non-Goals

- Training a frontier model.
- Hiding core scratch behavior behind Hugging Face model wrappers.
- Optimizing for benchmark leaderboard performance.
- Implementing CUDA kernels or distributed GPU training.
- Treating practical open-weight inference experiments as the scratch-core result.
