# Project Progress

Living coordination file for phase-based agent work.

## Active Phase

| Field | Value |
| --- | --- |
| Phase | PHASE-00B |
| Goal | Project Phase 0: repository and lab foundation |
| Branch | main |
| Worktree | /Users/keigoshimada/Documents/llm-from-scratch |
| Status | queued |

## Task Queue

### Open

- [ ] Execute PHASE-00B from `phase-plans/PHASE-00B-REPO-LAB-FOUNDATION.md`.
- [ ] Create the Python project skeleton, config system, dummy training loop, checkpoint format, and first tests.

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

## Phase Checklist

- [x] Acceptance criteria mapped to tasks for PHASE-00B.
- [ ] Required validation commands run.
- [ ] Evidence recorded.

## Future Backlog

- Train or select an English/Japanese tokenizer only after the micro character model proves the training loop.
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

## Phase Archive

_(none)_
