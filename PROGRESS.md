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

## Phase Archive

_(none)_
