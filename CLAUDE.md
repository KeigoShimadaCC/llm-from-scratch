# CLAUDE.md

## Project Workflow

This repo uses a phase-based agentic workflow for `KeigoGPT-Lab`, a Mac-local LLM-from-scratch lab.

Source of truth:

1. `docs/NORTH_STAR_LLM_FROM_SCRATCH_MAC.md`
2. `concept-and-ideas/**`
3. `phase-plans/**`
4. active phase plan
5. `PROGRESS.md`
6. generated run evidence under `runs/**`

## Rules

- Implement only the active phase.
- Prefer small, verifiable changes.
- Keep the scratch-core path self-owned: no pretrained model wrappers in core training.
- Run validation commands before claiming completion.
- Record evidence in `PROGRESS.md`.
- Treat agent output as a proposal until deterministic validation passes.
