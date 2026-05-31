# AGENTS.md

## Operating Rules

- Read `PROGRESS.md` first.
- Read `docs/NORTH_STAR_LLM_FROM_SCRATCH_MAC.md` before changing architecture, training, tokenizer, evaluation, or inference direction.
- Read the active phase plan in `phase-plans/` before editing.
- Keep changes inside the active phase allowed paths.
- Update `PROGRESS.md` with tasks, validation, and deferred backlog.
- Run required validation before claiming completion.
- Do not commit secrets, private datasets, checkpoints, or generated run artifacts.
- Do not modify generated runner evidence unless the phase explicitly requires it.

## Project Rules

- Core scratch path must implement the model, tokenizer path, training loop, evaluation, and inference owned by this repo.
- Do not use Hugging Face `AutoModelForCausalLM` or equivalent wrappers in the core scratch path.
- PyTorch is the first implementation target. MLX is a later Mac-native optimization phase.
- Every serious training run must be config-driven and produce reproducible logs.
- A tiny overfit test is required before trusting larger training runs.
