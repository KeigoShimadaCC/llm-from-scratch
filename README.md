# KeigoGPT-Lab

Mac-local LLM-from-scratch lab for building a small decoder-only language model from first principles.

The project target is educational but technically serious: implement the tokenizer path, model architecture, training loop, evaluation, and local inference path in this repo, while using PyTorch/MLX for numerical execution rather than hand-written kernels.

## Source Of Truth

- North Star: `docs/NORTH_STAR_LLM_FROM_SCRATCH_MAC.md`
- Phase plans: `phase-plans/`
- Runner state: `automation/phase-graph.json` and `automation/phase-state.json`
- Live coordination: `PROGRESS.md`

`NORTH_STAR.md` points to the full North Star doc for tools and agents that expect that filename at the repo root.

## From-Scratch Boundary

The scratch-core path owns:

- model architecture;
- tokenizer training/configuration path;
- dataset pipeline;
- training loop;
- evaluation harness;
- inference path;
- model weights trained from random initialization.

The core path must not hide behavior behind Hugging Face `AutoModelForCausalLM` or equivalent pretrained model wrappers. Reference implementations and practical open-weight experiments may exist later only as explicit comparison layers.

## Agentic Runner Setup

Build the local runner package before using `./bin/agentic` on a fresh clone:

```bash
pnpm --dir agentic-phase-runner-package install
pnpm --dir agentic-phase-runner-package run build
```

Inspect the current workflow:

```bash
./bin/agentic doctor --repo-root .
./bin/agentic status --repo-root .
```

Dry-run the first implementation phase:

```bash
./bin/agentic run --repo-root . --phase PHASE-00B --mode manual --dry-run
```

Run supervised agent execution only after reviewing the dry-run evidence:

```bash
./bin/agentic run --repo-root . --phase PHASE-00B --mode supervised --agents shell
```

## Validation Contract

Project phases use this baseline validation unless a phase plan adds more:

```bash
uv run pytest
uv run ruff check .
git diff --check
```

The current prep baseline does not implement Python source yet, so `PHASE-00B` is responsible for creating `pyproject.toml`, the package skeleton, and the first passing Python validation.

## Phase Roadmap

- `PHASE-00B`: repository and lab foundation
- `PHASE-01A`: MicroGPT character LM
- `PHASE-02A`: tokenizer and dataset pipeline
- `PHASE-03A`: core decoder-only Transformer
- `PHASE-04A`: tiny pretraining
- `PHASE-05A`: small practical model
- `PHASE-06A`: instruction tuning
- `PHASE-07A`: evaluation and failure analysis
- `PHASE-08A`: Mac-native inference
- `PHASE-09A`: final write-up and portfolio layer

## Artifact Policy

Private data, processed data, tokenized corpora, checkpoints, generated samples, and training runs must stay out of git. See `data/README.md` and `experiments/README.md`.
