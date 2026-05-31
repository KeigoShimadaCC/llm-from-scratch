# Project Progress

Living coordination file for phase-based agent work.

## Active Phase

| Field | Value |
| --- | --- |
| Phase | PHASE-02A |
| Goal | Tokenizer and dataset pipeline |
| Branch | main |
| Worktree | /Users/keigoshimada/Documents/llm-from-scratch |
| Status | queued |

## Task Queue

### Open

- [ ] Start PHASE-02A from `phase-plans/PHASE-02A-TOKENIZER-DATA-PIPELINE.md`.

### In Progress

_(none)_

### Done

- [x] Initialized the agentic phase-runner workflow files.
- [x] Configured Codex as the supervised shell-agent preset.
- [x] Replaced generic starter phase configuration with the Mac-local LLM project roadmap.
- [x] Realigned phase IDs to the North Star Phase 0-9 structure with PHASE-00B as the first implementation phase.
- [x] Added root docs for setup, data conventions, experiment artifacts, and North Star discovery.
- [x] Tightened phase-plan specs for artifact policy, data/tokenizer provenance, scale gates, evaluation assets, MLX inference, and final evidence mapping.
- [x] Clarified generated evidence versus forbidden paths, phase-specific validation expectations, and final North Star fallback positioning.
- [x] Added phase-specific validation command contracts, reviewed hybrid automerge policy, unattended decision policy, CI workflow, readiness command, and separate phase-acceptance versus merge gates.
- [x] Hardened migration, doctor, dry-run, and readiness checks so reviewed automation can reach 10/10 once the worktree is clean.
- [x] Added GitHub repository metadata from the North Star: description and project topics.
- [x] Added Python project metadata, config schema, deterministic seeding, checkpoint helpers, dummy training, placeholder CLIs, and foundation tests.
- [x] Completed PHASE-00B validation and recorded ignored run evidence.
- [x] Implemented PHASE-01A character tokenizer, context MLP training loop, deterministic generation, tests, prompt asset, mini report, and ignored overfit evidence.

## Phase Checklist

### PHASE-01A - MicroGPT Character LM

- [x] Accepted plan read before implementation.
- [x] North Star Phase 1 direction read before implementation.
- [x] Active phase plan read before implementation.
- [x] Character tokenizer utilities and roundtrip tests added.
- [x] Repo-authored synthetic fixture corpus documented.
- [x] Micro character model, config, and overfit training CLI added.
- [x] Seeded greedy and sampling generation CLI added.
- [x] Determinism and overfit tests added.
- [x] Fixed prompt asset or documented prompt list added.
- [x] Mini report added under `docs/`.
- [x] Required validation commands run.
- [x] Ignored run evidence recorded.

### PHASE-00B - Repository And Lab Foundation

- [x] Acceptance criteria mapped to tasks for PHASE-00B.
- [x] Accepted plan read before implementation.
- [x] North Star Phase 0 and training run artifact conventions read before implementation.
- [x] Python project metadata and lockfile added.
- [x] Importable package skeleton and CLI entrypoints added.
- [x] Config, seeding, checkpoint helpers added.
- [x] Dummy training command writes ignored run artifacts.
- [x] Foundation tests added.
- [x] Required validation commands run.
- [x] Evidence recorded.

## Future Backlog

- Train or select an English/Japanese tokenizer only after the micro character model proves the training loop.
- Expand the fixed prompt set in tokenizer and Transformer phases.
- Add richer generation metrics in PHASE-07A.
- Keep practical open-weight model experiments out of the scratch-core path unless a later phase explicitly creates a comparison branch.

## Validation Log

- 2026-05-31: `./bin/agentic doctor --repo-root .` passed after PHASE-00B graph realignment.
- 2026-05-31: `./bin/agentic status --repo-root .` reported PHASE-00B as next runnable with 10 queued phases.
- 2026-05-31: `./bin/agentic run --repo-root . --phase PHASE-00B --mode manual --dry-run` generated the dry-run evidence bundle without invoking agents.
- 2026-05-31: `pnpm --dir agentic-phase-runner-package run typecheck` passed.
- 2026-05-31: `pnpm --dir agentic-phase-runner-package run test` passed: 14 files, 62 tests.
- 2026-05-31: `git diff --check` passed.
- 2026-05-31: `./bin/agentic doctor --repo-root .` passed after North Star phase-plan gap tightening.
- 2026-05-31: `./bin/agentic status --repo-root .` still reports PHASE-00B as next runnable.
- 2026-05-31: `./bin/agentic run --repo-root . --phase PHASE-00B --mode manual --dry-run` completed after phase-plan gap tightening.
- 2026-05-31: `git diff --check` passed after phase-plan gap tightening.
- 2026-05-31: `./bin/agentic doctor --repo-root .` passed after final phase-plan tightening tweaks.
- 2026-05-31: `./bin/agentic status --repo-root .` still reports PHASE-00B as current and next runnable.
- 2026-05-31: `./bin/agentic run --repo-root . --phase PHASE-00B --mode manual --dry-run` completed after final phase-plan tightening tweaks.
- 2026-05-31: `git diff --check` passed after final phase-plan tightening tweaks.
- 2026-05-31: `pnpm --dir agentic-phase-runner-package run typecheck` passed after automation-readiness hardening changes.
- 2026-05-31: `pnpm --dir agentic-phase-runner-package run test` passed after automation-readiness hardening changes: 16 files, 68 tests.
- 2026-05-31: `pnpm --dir agentic-phase-runner-package run build` passed after automation-readiness hardening changes.
- 2026-05-31: `./bin/agentic doctor --repo-root .` passed with no migration warnings after reviewed automerge policy handling.
- 2026-05-31: `./bin/agentic run --repo-root . --phase PHASE-00B --mode auto --preset codex --plan-approval auto --dry-run` completed and wrote ignored evidence under `runs/phase-runner/PHASE-00B/2026-05-31T13:57:25Z`.
- 2026-05-31: `./bin/agentic readiness --repo-root . --target phase00b-auto` and `--target unattended` reported 7/10 in sandbox because the worktree was dirty and GitHub network/auth checks could not complete.
- 2026-05-31: Escalated `./bin/agentic readiness --repo-root . --target phase00b-auto` reported 9/10: GitHub auth and remote reachability passed, with only the current dirty worktree blocking 10/10.
- 2026-05-31: `git diff --check` passed after automation-readiness hardening changes.
- 2026-05-31: After committing automation-readiness hardening, escalated `./bin/agentic readiness --repo-root . --target phase00b-auto` reported 10/10.
- 2026-05-31: After committing automation-readiness hardening, escalated `./bin/agentic readiness --repo-root . --target unattended` reported 10/10.
- 2026-05-31: First live `PHASE-00B` auto run blocked at plan acceptance because the planner prompt did not require `allowedPaths` or `acceptanceCriteriaCovered` in task JSON; tightened planner templates and reset `PHASE-00B` to queued before rerun.
- 2026-05-31: Escalated `gh repo view KeigoShimadaCC/llm-from-scratch --json description,homepageUrl,repositoryTopics,url` verified the GitHub repo description and topics after metadata update.
- 2026-05-31: Second live `PHASE-00B` auto run passed plan acceptance but executor hit the 5-minute inactivity timeout after starting implementation; increased executor/rechecker inactivity timeouts locally before resume.
- 2026-05-31: Nested executor remained stalled after safer streaming resume, so PHASE-00B implementation was steered directly in the main workspace while preserving accepted-plan evidence under `runs/phase-runner/PHASE-00B/2026-05-31T14:18:34Z/`.
- 2026-05-31: `uv run pytest` passed for PHASE-00B: 5 tests.
- 2026-05-31: `uv lock --check` passed after adding `pyproject.toml` and `uv.lock`.
- 2026-05-31: `uv run ruff check .` passed.
- 2026-05-31: `git diff --check` passed.
- 2026-05-31: `pnpm --dir agentic-phase-runner-package run typecheck`, `test`, and `build` passed after the runner output-capture steering change.
- 2026-05-31: `uv run python -m train.dummy --config configs/dummy.yaml --run-name phase00b_smoke` passed and generated ignored evidence under `experiments/runs/20260531T145036Z_phase00b_smoke/`.
- 2026-05-31: `uv run python -m kgpt.env_check --optional-mps` passed with Python 3.12.11, Torch 2.12.0, CPU available, and MPS available.
- 2026-05-31: `./bin/agentic doctor --repo-root .` passed after PHASE-00B implementation.
- 2026-05-31: `./bin/agentic status --repo-root .` reported PHASE-01A as the next runnable phase after phase state was advanced.
- 2026-06-01: `uv run pytest tests/test_char_tokenizer.py tests/test_micro_char_training.py` passed for PHASE-01A focused tokenizer, overfit, and deterministic generation checks: 3 tests.
- 2026-06-01: `uv run pytest` passed for PHASE-01A: 8 tests.
- 2026-06-01: `uv run ruff check .` passed after fixing initial style findings.
- 2026-06-01: `git diff --check` passed.
- 2026-06-01: `uv run python -m train.micro_char --config configs/micro_char.yaml --max-steps 200 --run-name phase01a_overfit_smoke` passed and generated ignored evidence under `experiments/runs/phase01a_overfit_smoke/`; train loss went from 2.714604377746582 to 0.00000252200834438554.
- 2026-06-01: `uv run python -m inference.generate_char --checkpoint experiments/runs/phase01a_overfit_smoke/checkpoint_last.pt --prompt hello --seed 123 --max-new-tokens 32` passed and generated `hello microgpt.\nhello microgpt.\nhello`.

## Phase Archive

### PHASE-01A - MicroGPT Character LM

Status: complete
Completed: 2026-06-01
Evidence:
- Accepted automation plan: `runs/phase-runner/PHASE-01A/2026-05-31T16:09:24Z/accepted-plan/accepted-plan.json`
- Ignored overfit run: `experiments/runs/phase01a_overfit_smoke/`
- Mini report: `docs/phase01a_micro_char_report.md`
- Fixed prompt file: `eval/char_prompts.json`
- Validation commands: `uv run pytest`, `uv run ruff check .`, `git diff --check`, `uv run python -m train.micro_char --config configs/micro_char.yaml --max-steps 200 --run-name phase01a_overfit_smoke`, `uv run python -m inference.generate_char --checkpoint experiments/runs/phase01a_overfit_smoke/checkpoint_last.pt --prompt hello --seed 123 --max-new-tokens 32`

### PHASE-00B - Repository And Lab Foundation

Status: complete
Completed: 2026-05-31
Evidence:
- Accepted automation plan: `runs/phase-runner/PHASE-00B/2026-05-31T14:18:34Z/accepted-plan/accepted-plan.json`
- Ignored dummy run: `experiments/runs/20260531T145036Z_phase00b_smoke/`
- Validation commands: `uv run pytest`, `uv run ruff check .`, `git diff --check`, `uv run python -m train.dummy --config configs/dummy.yaml --run-name phase00b_smoke`, `uv run python -m kgpt.env_check --optional-mps`
