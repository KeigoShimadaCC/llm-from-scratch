# Project Progress

Living coordination file for phase-based agent work.

## Active Phase

| Field | Value |
| --- | --- |
| Phase | CROSS-PHASE-GAP-FIX |
| Goal | Repair phase-plan implementation gaps in reproducible evidence, audits, docs, and inference CLI consistency |
| Branch | main |
| Worktree | /Users/keigoshimada/Documents/llm-from-scratch |
| Status | complete |

## Task Queue

### Open

- [x] None.

### In Progress

- [ ] None.

### Done

- [x] Merged cross-phase gap fix via PR #10 at merge commit `26c076ac0f1a05066b11336cb3e9255b2cf2722c`.
- [x] Full validation passed after final fixes: `uv run pytest` (41 tests), `uv run ruff check .`, `git diff --check`, `uv run python -m eval.audit_claims --doc docs/FINAL_WRITEUP.md --output docs/claim_evidence_audit.md`, and `uv run python -m eval.check_repro_commands --doc docs/COMMAND_INDEX.md`.
- [x] Artifact policy check passed: `git ls-files experiments/runs data/tokenized '*.pt' '*.safetensors'` returned no tracked artifacts.
- [x] Phase-plan completion audit passed for the identified gaps: no stale PHASE-01A command, no fake PHASE-08A run dir, PHASE-04A/06A ignored run manifests include `eval_report.md`, PHASE-05A report includes 30M checkpoint/sample evidence, and PHASE-07A live-evaluates all regenerated checkpoint rows.
- [x] Regenerated tokenizer artifacts, token batches, PHASE-03A micro checkpoint, PHASE-04A tiny run, PHASE-05A 30M run, PHASE-06A SFT run, PHASE-07A live comparison reports, and PHASE-08A benchmark evidence; generated checkpoints and run dirs remain ignored.
- [x] Confirmed PHASE-07A comparison now live-evaluates all four listed checkpoints and reports no missing ignored artifacts after local regeneration.
- [x] Patched command/artifact indexes, final audit scripts, run-dir eval report manifest updates, PHASE-05A 30M checkpoint/sample evidence rendering, and PHASE-08A chat CLI option parity.
- [x] Focused validation passed: `uv run pytest tests/test_final_audits.py tests/test_inference_runtime.py`, `uv run ruff check ...`, and `uv run python -m eval.check_repro_commands --doc docs/COMMAND_INDEX.md`.
- [x] Started cross-phase gap-fix branch from clean `main`; read `PROGRESS.md`, North Star direction, PHASE-00A global invariants, and relevant phase plans before edits.
- [x] Merged PHASE-09A via PR #9 at merge commit `d8dc7da9fd18fc75d678653755ef71b2773ef246`.
- [x] Implemented PHASE-09A final write-up, README navigation, command index, artifact index, model/data cards, claim audit, and command-index audit.
- [x] Validated PHASE-09A locally with final audit commands, full tests, lint, and whitespace checks.
- [x] Completed PHASE-09A plan audit; no blocking implementation gaps remain. The final positioning is explicitly partial with documented fallback.
- [x] Merged PHASE-08A via PR #8 at merge commit `477ca2850ed889288da1cec9162752cb0ca0fdab`.
- [x] Implemented PHASE-08A completion/chat CLI, sampling controls, KV-cache parity, benchmark report, and MLX deferral evidence.
- [x] Validated PHASE-08A locally with generation smoke, KV-cache parity, benchmark, full tests, lint, and whitespace checks.
- [x] Completed PHASE-08A plan audit; no blocking implementation gaps remain. MLX is explicitly deferred, not claimed as achieved.
- [x] Merged PHASE-07A via PR #7 at merge commit `af0b07bed26e67bbe788b0eec4fb93566e42543f`.
- [x] Implemented PHASE-07A checkpoint evaluation, fixed prompt consolidation, failure taxonomy, and comparison reports.
- [x] Validated PHASE-07A locally with report generation, checkpoint comparison, full tests, lint, and whitespace checks.
- [x] Completed PHASE-07A plan audit; no blocking implementation gaps remain. Larger checkpoint rows are summary-only until ignored local artifacts are regenerated.
- [x] Merged PHASE-06A via PR #6 at merge commit `4f3fd572e2f8ba0457445742ba08e083d658c97c`.
- [x] Implemented PHASE-06A SFT config, prompt template, dataset manifest, response-only masking, trainer, and comparison report.
- [x] Validated PHASE-06A locally with SFT smoke and base-vs-SFT fixed probes.
- [x] Merged PHASE-05A via PR #5 at merge commit `535ebf9b982cec4e6cb6aa8bc86a663408983316`.
- [x] Implemented PHASE-05A configs, dry-run/profile tooling, scaling report, and 30M+ local training evidence.
- [x] Validated PHASE-05A locally with 30M dry-run/resume validation and scaling report generation.
- [x] Merged PHASE-04A via PR #4 at merge commit `988fa5abfb1aa344cfe526de5a5bb0629c2b6b36`.
- [x] Implemented PHASE-04A from the phase plan after runner planning stalled before producing an accepted plan.
- [x] Validated PHASE-04A tiny pretraining, fixed-prompt report generation, and resume probe locally.
- [x] Merged PHASE-03A via PR #3 at merge commit `d672096c7ebdbee1e09bf85431380e5105a4b9bc`.
- [x] Implemented PHASE-03A from accepted plan `runs/phase-runner/PHASE-03A/2026-05-31T17:12:39Z/accepted-plan/accepted-plan.json`.
- [x] Validated PHASE-03A Transformer architecture, smoke training, and generation commands locally.
- [x] Validated PHASE-02A implementation from the accepted plan at `runs/phase-runner/PHASE-02A/2026-05-31T16:41:04Z/accepted-plan/accepted-plan.json`.
- [x] Merged PHASE-02A via PR #2 at merge commit `a1295e568294980f11dff5d7590ce5890d1b9d24`.
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

### PHASE-09A - Final Write-Up And Portfolio Layer

- [x] North Star final positioning read before implementation.
- [x] Active phase plan read before implementation.
- [x] Final technical write-up added.
- [x] README navigation updated.
- [x] Claim-to-evidence table added and audited.
- [x] Artifact index added.
- [x] Reproducible command index added and checked.
- [x] Model/data cards added or linked.
- [x] What changed between phases section added.
- [x] Final positioning distinguishes achieved work from documented fallback.
- [x] Required validation commands run.

### PHASE-08A - Mac-Native Inference

- [x] North Star inference direction read before implementation.
- [x] Active phase plan read before implementation.
- [x] Inference smoke and benchmark configs added.
- [x] Completion CLI supports checkpoint/config/tokenizer/prompt/max tokens/temperature/top-k/top-p/seed/device/dtype/stop/output options.
- [x] Chat/instruction-format CLI added.
- [x] Greedy, temperature, top-k, top-p, deterministic seeding, stop behavior, and max-new-token behavior tested or documented.
- [x] KV-cache support and parity validation added or formally deferred.
- [x] CPU/MPS/MLX benchmark protocol and report added.
- [x] MLX comparison implemented or formally deferred with blocker evidence.
- [x] Local inference guide/model card added.
- [x] Required validation commands run.
- [x] Ignored inference evidence recorded.

### PHASE-07A - Evaluation And Failure Analysis

- [x] North Star evaluation direction read before implementation.
- [x] Active phase plan read before implementation.
- [x] Fixed qualitative prompt set consolidated across earlier phases.
- [x] Checkpoint-compatible evaluation CLI added.
- [x] Metric definitions, report schema, and failure taxonomy documented.
- [x] Leakage and memorization probes included or explicitly blocked by missing data evidence.
- [x] Cross-checkpoint comparison report generated.
- [x] Required validation commands run.
- [x] Ignored eval evidence recorded if generated.

### PHASE-06A - Instruction Tuning

- [x] North Star Phase 6 direction read before implementation.
- [x] Active phase plan read before implementation.
- [x] Prompt template version and SFT configs added.
- [x] Instruction dataset source/license, split, dedup, contamination notes, and fixed eval set documented.
- [x] SFT data loading and response-only loss masking implemented.
- [x] SFT training path produces ignored checkpoint and manifest.
- [x] Base-vs-SFT comparison report implemented and generated.
- [x] Prompt masking and dataset tests added.
- [x] Required validation commands run.
- [x] Ignored SFT evidence recorded.

### PHASE-05A - Small Practical Model

- [x] North Star Phase 5 direction read before implementation.
- [x] Active phase plan read before implementation.
- [x] `kgpt-30m` config and stretch configs with computed parameter counts added.
- [x] Final tokenizer decision and data mixture manifest documented.
- [x] 30M dry-run, resume validation, profiling, and scaling comparison commands added.
- [x] At least one 30M+ local run completed or infeasibility documented with fallback evidence.
- [x] Scaling report compares micro, tiny, and 30M+ behavior.
- [x] Stretch `kgpt-50m`/`kgpt-100m` decision recorded.
- [x] Required validation commands run.
- [x] Ignored 30M+ evidence recorded.

### PHASE-04A - Tiny Pretraining Run

- [x] North Star Phase 4 direction read before implementation.
- [x] Active phase plan read before implementation.
- [x] Runner attempt preserved as evidence under `runs/phase-runner/PHASE-04A/2026-05-31T17:35:44Z/`.
- [x] Tiny pretraining dataset config with source/license, split, dedup, and leakage notes added.
- [x] 5M-20M parameter tiny model config, optimizer, scheduler, gradient accumulation, cadence, and budget added.
- [x] Token-level pretraining loop, metrics logging, best-checkpoint selection, samples, and resume behavior added.
- [x] Fixed-prompt eval config and report generator added.
- [x] Resume or checkpoint behavior tests added.
- [x] Required validation commands run.
- [x] Ignored tiny-run evidence recorded.

### PHASE-03A - Core Decoder-Only Transformer

- [x] Accepted plan read before implementation.
- [x] North Star Phase 3 direction read before implementation.
- [x] Active phase plan read before implementation.
- [x] Transformer config, device, dtype, and sampling policy added.
- [x] Decoder-only Transformer, causal attention, shifted loss, parameter counting, and tied embeddings implemented.
- [x] Autoregressive generation path added.
- [x] Transformer smoke trainer integrated with PHASE-02A token batches or documented synthetic fallback.
- [x] Architecture invariant tests added.
- [x] Architecture note added under `docs/`.
- [x] Required validation commands run.
- [x] Ignored micro-run evidence recorded.

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

### PHASE-02A - Tokenizer And Dataset Pipeline

- [x] Accepted plan read before implementation.
- [x] North Star Phase 2 direction read before implementation.
- [x] Active phase plan read before implementation.
- [x] Bilingual tokenizer config and smoke tokenized-data config added.
- [x] Tokenizer candidates, vocabulary sweep, and report generation implemented.
- [x] Data manifest schema, preprocessing, dedup, split generation, token writing, and metadata sidecar implemented.
- [x] Token-level language-model batch sampler implemented.
- [x] Roundtrip, split-integrity, leakage-check, dedup-smoke, batch-shape, and small-model-consumption tests added.
- [x] Required validation commands run.
- [x] Recheck validation rerun against the PHASE-02A plan, accepted plan, executor report, changed files, and generated evidence.
- [x] Deferred tokenizer choice and larger-data backlog recorded.

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
- Revisit the final tokenizer choice before larger PHASE-05A scale-up if the PHASE-02A synthetic bilingual smoke corpus is insufficient evidence.

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
- 2026-06-01: PHASE-02A implementation started from the accepted plan; validation not run yet.
- 2026-06-01: PHASE-02A focused tests passed: `uv run pytest tests/test_byte_bpe_tokenizer.py tests/test_tokenizer_report.py tests/test_token_data_pipeline.py`.
- 2026-06-01: PHASE-02A smoke commands passed and generated `docs/tokenizer_report.md`, `docs/phase02a_data_manifest.json`, and ignored token artifacts under `data/tokenized/`.
- 2026-06-01: `uv run pytest` passed for PHASE-02A: 11 tests.
- 2026-06-01: `uv run ruff check .` passed.
- 2026-06-01: `git diff --check` passed.
- 2026-06-01: `uv run python -m tokenizer.train_report --config configs/tokenizer_bilingual.yaml --output docs/tokenizer_report.md` passed and generated a 377-token byte-level BPE tokenizer report.
- 2026-06-01: `uv run python -m train.sample_batches --config configs/tokenized_smoke.yaml --max-batches 2` passed with input/target shapes `[4, 16]`, logits `[4, 16, 377]`, and ignored split artifacts under `data/tokenized/phase02a_smoke/`.
- 2026-06-01: Recheck reran `uv run pytest`, `uv run ruff check .`, `git diff --check`, `uv run python -m tokenizer.train_report --config configs/tokenizer_bilingual.yaml --output docs/tokenizer_report.md`, and `uv run python -m train.sample_batches --config configs/tokenized_smoke.yaml --max-batches 2`; all passed. Metadata evidence confirmed `train`/`validation` splits, `uint16` token files, vocab size 377, zero leakage overlap, and one duplicate removed.
- 2026-06-01: PHASE-02A PR #2 passed GitHub CI and merged to `main` at `a1295e568294980f11dff5d7590ce5890d1b9d24`. The runner final gate had falsely blocked because it checked remote status before GitHub registered checks; automation was patched to poll `statusCheckRollup` after an initial no-checks result.
- 2026-06-01: PHASE-03A focused tests passed: `uv run pytest tests/test_transformer.py -q` -> 7 passed.
- 2026-06-01: `uv run pytest` passed for PHASE-03A: 18 tests.
- 2026-06-01: `uv run ruff check .` passed.
- 2026-06-01: `git diff --check` passed.
- 2026-06-01: `uv run python -m train.transformer_smoke --config configs/transformer_micro.yaml --max-steps 20` passed and generated ignored evidence under `experiments/runs/phase03a_transformer_micro/`; train loss went from 22.25092887878418 to 1.0929747819900513 with 29,728 parameters.
- 2026-06-01: `uv run python -m inference.generate --config configs/transformer_micro.yaml --prompt hello --max-new-tokens 16 --seed 123` passed from the Transformer smoke checkpoint and generated 16 new tokens.
- 2026-06-01: PHASE-03A PR #3 passed GitHub CI and merged to `main` at `d672096c7ebdbee1e09bf85431380e5105a4b9bc`.
- 2026-06-01: PHASE-04A live runner attempt stalled in planning with zero-byte planner logs under `runs/phase-runner/PHASE-04A/2026-05-31T17:35:44Z/`; implementation proceeded manually from the phase plan.
- 2026-06-01: PHASE-04A focused tests passed: `uv run pytest tests/test_pretrain.py -q` -> 3 passed.
- 2026-06-01: `uv run pytest` passed for PHASE-04A: 21 tests.
- 2026-06-01: `uv run ruff check .` passed.
- 2026-06-01: `git diff --check` passed.
- 2026-06-01: `uv run python -m train.pretrain --config configs/kgpt_tiny.yaml --max-steps 200 --run-name phase04a_tiny_smoke` passed and generated ignored evidence under `experiments/runs/phase04a_tiny_smoke/`; parameter count was 5,633,536 and validation loss improved from 164.59511184692383 to 4.059329509735107.
- 2026-06-01: `uv run python -m eval.report --config configs/eval_fixed_prompts.yaml --checkpoint experiments/runs/phase04a_tiny_smoke/checkpoint_last.pt --output docs/phase04a_tiny_report.md` passed and wrote the fixed-prompt report.
- 2026-06-01: `uv run python -m train.pretrain --config configs/kgpt_tiny.yaml --max-steps 8 --run-name phase04a_runtime_probe --resume` passed, proving checkpoint resume from the earlier 5-step probe.
- 2026-06-01: PHASE-04A PR #4 passed GitHub CI and merged to `main` at `988fa5abfb1aa344cfe526de5a5bb0629c2b6b36`.
- 2026-06-01: PHASE-05A focused tests passed: `uv run pytest tests/test_pretrain.py -q` -> 4 passed.
- 2026-06-01: `uv run python -m train.pretrain --config configs/kgpt_30m.yaml --dry-run --validate-resume` passed with 31,734,272 parameters and resume validation.
- 2026-06-01: `uv run python -m train.pretrain --config configs/kgpt_30m.yaml --max-steps 40 --run-name phase05a_kgpt30m_smoke` passed and generated ignored evidence under `experiments/runs/phase05a_kgpt30m_smoke/`; validation loss improved from 316.75299072265625 to 11.464725255966187.
- 2026-06-01: `uv run python -m train.pretrain --config configs/kgpt_50m.yaml --dry-run` passed with 59,345,280 parameters.
- 2026-06-01: `uv run python -m train.pretrain --config configs/kgpt_100m.yaml --dry-run` passed with 113,721,600 parameters.
- 2026-06-01: `uv run pytest` passed for PHASE-05A: 22 tests.
- 2026-06-01: `uv run ruff check .` passed.
- 2026-06-01: `git diff --check` passed.
- 2026-06-01: `uv run python -m eval.compare_runs --manifest docs/phase05a_scaling_manifest.json --output docs/phase05a_scaling_report.md` passed and confirmed a trained 30M+ run in the scaling manifest.
- 2026-06-01: PHASE-05A PR #5 passed GitHub CI and merged to `main` at `535ebf9b982cec4e6cb6aa8bc86a663408983316`.
- 2026-06-01: PHASE-06A focused tests passed: `uv run pytest tests/test_sft.py -q` -> 3 passed.
- 2026-06-01: `uv run pytest` passed for PHASE-06A: 25 tests.
- 2026-06-01: `uv run ruff check .` passed.
- 2026-06-01: `git diff --check` passed.
- 2026-06-01: `uv run python -m train.sft --config configs/sft_smoke.yaml --max-steps 50 --run-name phase06a_sft_smoke` passed and generated ignored SFT evidence under `experiments/runs/phase06a_sft_smoke/`; held-out validation loss regressed from 32.469276428222656 to 63.17095756530762.
- 2026-06-01: `uv run python -m eval.sft_compare --config configs/sft_eval.yaml --output docs/phase06a_sft_eval.md` passed; fixed-probe response loss improved from 35.40309000015259 to 0.0000020305075345561363.
- 2026-06-01: PHASE-06A PR #6 passed GitHub CI and merged to `main` at `4f3fd572e2f8ba0457445742ba08e083d658c97c`.
- 2026-06-01: PHASE-07A validation passed: `uv run pytest` (29 tests), `uv run ruff check .`, `git diff --check`, `uv run python -m eval.report --config configs/eval_fixed_prompts.yaml --output docs/phase07a_eval_report.md`, and `uv run python -m eval.compare_checkpoints --manifest docs/checkpoint_manifest.json --output docs/phase07a_checkpoint_comparison.md`. Both PHASE-07A reports completed with one live micro checkpoint row from `experiments/runs/phase03a_transformer_micro/` and three summary-only rows for missing ignored tiny/30M/SFT artifacts.
- 2026-06-01: PHASE-07A PR #7 passed GitHub CI and merged to `main` at `af0b07bed26e67bbe788b0eec4fb93566e42543f`.
- 2026-06-01: PHASE-08A validation passed: `uv run pytest` (34 tests), `uv run ruff check .`, `git diff --check`, `uv run python -m inference.generate --config configs/inference_smoke.yaml --prompt hello --max-new-tokens 16 --seed 123`, `uv run python -m inference.kv_cache_parity --config configs/inference_smoke.yaml`, and `uv run python -m inference.benchmark --config configs/inference_benchmark.yaml --max-new-tokens 32 --output docs/phase08a_benchmark.md`. The local benchmark measured CPU and PyTorch MPS and formally deferred MLX with blocker evidence.
- 2026-06-01: PHASE-08A PR #8 passed GitHub CI and merged to `main` at `477ca2850ed889288da1cec9162752cb0ca0fdab`.
- 2026-06-01: PHASE-09A validation passed: `uv run pytest` (34 tests), `uv run ruff check .`, `git diff --check`, `uv run python -m eval.audit_claims --doc docs/FINAL_WRITEUP.md --output docs/claim_evidence_audit.md` (11 claims, 0 unsupported), and `uv run python -m eval.check_repro_commands --doc docs/COMMAND_INDEX.md` (18 required command snippets, 0 missing).
- 2026-06-01: PHASE-09A PR #9 passed GitHub CI and merged to `main` at `d8dc7da9fd18fc75d678653755ef71b2773ef246`. The final phase state was updated so all 10 phases are complete.
- 2026-06-01: Cross-phase gap-fix regenerated ignored evidence with tokenizer/report, sample batch, PHASE-03A smoke, PHASE-04A tiny pretraining, PHASE-05A 30M smoke, PHASE-06A SFT, PHASE-07A live eval/comparison, and PHASE-08A benchmark commands from the remediation plan.
- 2026-06-01: Cross-phase gap-fix validation passed: `uv run pytest` (41 tests), `uv run ruff check .`, `git diff --check`, `uv run python -m eval.audit_claims --doc docs/FINAL_WRITEUP.md --output docs/claim_evidence_audit.md` (11 claims, 0 unsupported), `uv run python -m eval.check_repro_commands --doc docs/COMMAND_INDEX.md` (26 exact commands, 0 missing, 0 stale tokens), and `git ls-files experiments/runs data/tokenized '*.pt' '*.safetensors'` returned no tracked artifacts.
- 2026-06-01: Cross-phase gap-fix PR #10 passed GitHub CI and merged to `main` at `26c076ac0f1a05066b11336cb3e9255b2cf2722c`.

## Phase Archive

### CROSS-PHASE-GAP-FIX - Evidence And Automation Readiness Repair

Status: complete
Started: 2026-06-01
Evidence:
- PR: #10
- Merge commit: `26c076ac0f1a05066b11336cb3e9255b2cf2722c`
- Corrected command index: `docs/COMMAND_INDEX.md`
- Corrected artifact index: `docs/ARTIFACT_INDEX.md`
- Stricter final audits: `eval/check_repro_commands.py`, `eval/audit_claims.py`, `tests/test_final_audits.py`
- Run-local eval report manifest support: `eval/report.py`, `eval/sft_compare.py`
- 30M checkpoint/sample evidence renderer: `eval/compare_runs.py`, `docs/phase05a_scaling_report.md`
- PHASE-08A chat CLI parity: `inference/chat.py`, `docs/phase08a_inference_guide.md`
- Live regenerated reports: `docs/phase04a_tiny_report.md`, `docs/phase06a_sft_eval.md`, `docs/phase07a_eval_report.md`, `docs/phase07a_checkpoint_comparison.md`, `docs/phase08a_benchmark.md`
- Ignored regenerated evidence: `data/tokenized/`, `experiments/runs/phase03a_transformer_micro/`, `experiments/runs/phase04a_tiny_smoke/`, `experiments/runs/phase05a_kgpt30m_smoke/`, `experiments/runs/phase06a_sft_smoke/`
- Validation commands: `uv run pytest`, `uv run ruff check .`, `git diff --check`, phase-specific regeneration commands from `docs/COMMAND_INDEX.md`, final claim audit, command-index audit, and tracked-artifact check.
- Result note: This fixes the audited implementation gaps without changing the final project positioning; MLX remains formally deferred and the North Star remains partially achieved with documented fallback.

### PHASE-09A - Final Write-Up And Portfolio Layer

Status: complete
Completed: 2026-06-01
Evidence:
- PR: #9
- Merge commit: `d8dc7da9fd18fc75d678653755ef71b2773ef246`
- Final write-up: `docs/FINAL_WRITEUP.md`
- README navigation: `README.md`
- Command index: `docs/COMMAND_INDEX.md`
- Artifact index: `docs/ARTIFACT_INDEX.md`
- Model/data cards: `docs/MODEL_CARD.md`, `docs/DATA_CARD.md`
- Claim audit: `docs/claim_evidence_audit.md`
- Audit CLIs: `eval/audit_claims.py`, `eval/check_repro_commands.py`
- Validation commands: `uv run pytest`, `uv run ruff check .`, `git diff --check`, `uv run python -m eval.audit_claims --doc docs/FINAL_WRITEUP.md --output docs/claim_evidence_audit.md`, `uv run python -m eval.check_repro_commands --doc docs/COMMAND_INDEX.md`
- Result note: Final positioning is `North Star partially achieved with documented fallback`; MLX inference remains deferred and 50M/100M stretch models are dry-run targets.

### PHASE-08A - Mac-Native Inference

Status: complete
Completed: 2026-06-01
Evidence:
- PR: #8
- Merge commit: `477ca2850ed889288da1cec9162752cb0ca0fdab`
- Inference configs: `configs/inference_smoke.yaml`, `configs/inference_benchmark.yaml`
- Completion/chat/runtime CLIs: `inference/generate.py`, `inference/chat.py`, `inference/runtime.py`
- KV-cache and benchmark CLIs: `inference/kv_cache_parity.py`, `inference/benchmark.py`
- Transformer KV-cache support: `kgpt/transformer.py`
- Guide and reports: `docs/phase08a_inference_guide.md`, `docs/phase08a_model_loading_report.md`, `docs/phase08a_benchmark.md`
- Evidence artifacts: `docs/phase08a_generation_example.json`, `docs/phase08a_kv_cache_parity.json`
- MLX deferral: `docs/phase08a_mlx_deferral.md`
- Tests: `tests/test_transformer.py`, `tests/test_inference_runtime.py`
- Ignored inference artifacts: `experiments/runs/phase03a_transformer_micro/`, `data/tokenized/**`
- Validation commands: `uv run pytest`, `uv run ruff check .`, `git diff --check`, `uv run python -m inference.generate --config configs/inference_smoke.yaml --prompt hello --max-new-tokens 16 --seed 123`, `uv run python -m inference.kv_cache_parity --config configs/inference_smoke.yaml`, `uv run python -m inference.benchmark --config configs/inference_benchmark.yaml --max-new-tokens 32 --output docs/phase08a_benchmark.md`
- Result note: CPU and PyTorch MPS benchmark rows were measured locally; MLX is explicitly deferred and must not be described as a completed PyTorch-vs-MLX comparison.

### PHASE-07A - Evaluation And Failure Analysis

Status: complete
Completed: 2026-06-01
Evidence:
- PR: #7
- Merge commit: `af0b07bed26e67bbe788b0eec4fb93566e42543f`
- Eval config: `configs/eval_fixed_prompts.yaml`
- Checkpoint manifest: `docs/checkpoint_manifest.json`
- Eval schema and taxonomy: `docs/phase07a_eval_schema.md`
- Eval report: `docs/phase07a_eval_report.md`
- Comparison report: `docs/phase07a_checkpoint_comparison.md`
- Evaluation implementation: `eval/checkpoint_eval.py`, `eval/report.py`, `eval/compare_checkpoints.py`
- Tests: `tests/test_checkpoint_eval.py`
- Ignored live micro checkpoint evidence: `experiments/runs/phase03a_transformer_micro/`
- Validation commands: `uv run pytest`, `uv run ruff check .`, `git diff --check`, `uv run python -m eval.report --config configs/eval_fixed_prompts.yaml --output docs/phase07a_eval_report.md`, `uv run python -m eval.compare_checkpoints --manifest docs/checkpoint_manifest.json --output docs/phase07a_checkpoint_comparison.md`
- Result note: PHASE-07A reports contain one live micro row and summary-only rows for tiny, 30M, and SFT when ignored checkpoints are absent; reproduce commands are listed in `docs/checkpoint_manifest.json`.

### PHASE-06A - Instruction Tuning

Status: complete
Completed: 2026-06-01
Evidence:
- PR: #6
- Merge commit: `4f3fd572e2f8ba0457445742ba08e083d658c97c`
- SFT config: `configs/sft_smoke.yaml`
- SFT eval config: `configs/sft_eval.yaml`
- Data manifest: `docs/phase06a_instruction_data_manifest.json`
- SFT eval report: `docs/phase06a_sft_eval.md`
- SFT implementation: `kgpt/sft.py`, `train/sft.py`
- Comparison implementation: `eval/sft_compare.py`
- Ignored SFT run: `experiments/runs/phase06a_sft_smoke/`
- Validation commands: `uv run pytest`, `uv run ruff check .`, `git diff --check`, `uv run python -m train.sft --config configs/sft_smoke.yaml --max-steps 50 --run-name phase06a_sft_smoke`, `uv run python -m eval.sft_compare --config configs/sft_eval.yaml --output docs/phase06a_sft_eval.md`
- Result note: fixed-probe response loss improved after SFT while held-out validation regressed; claims are limited to the narrow fixed command probes documented in `docs/phase06a_sft_eval.md`.

### PHASE-05A - Small Practical Model

Status: complete
Completed: 2026-06-01
Evidence:
- PR: #5
- Merge commit: `535ebf9b982cec4e6cb6aa8bc86a663408983316`
- 30M config: `configs/kgpt_30m.yaml`
- Stretch configs: `configs/kgpt_50m.yaml`, `configs/kgpt_100m.yaml`
- Tokenizer decision: `docs/phase05a_tokenizer_decision.md`
- Data mixture manifest: `docs/phase05a_data_mixture_manifest.json`
- Mac profile: `docs/phase05a_mac_profile.md`
- Scaling manifest/report: `docs/phase05a_scaling_manifest.json`, `docs/phase05a_scaling_report.md`
- Ignored 30M run: `experiments/runs/phase05a_kgpt30m_smoke/`
- Validation commands: `uv run pytest`, `uv run ruff check .`, `git diff --check`, `uv run python -m train.pretrain --config configs/kgpt_30m.yaml --dry-run --validate-resume`, `uv run python -m eval.compare_runs --manifest docs/phase05a_scaling_manifest.json --output docs/phase05a_scaling_report.md`
- Extra local evidence: `uv run python -m train.pretrain --config configs/kgpt_30m.yaml --max-steps 40 --run-name phase05a_kgpt30m_smoke`, `uv run python -m train.pretrain --config configs/kgpt_50m.yaml --dry-run`, `uv run python -m train.pretrain --config configs/kgpt_100m.yaml --dry-run`

### PHASE-04A - Tiny Pretraining Run

Status: complete
Completed: 2026-06-01
Evidence:
- Runner attempt: `runs/phase-runner/PHASE-04A/2026-05-31T17:35:44Z/`
- PR: #4
- Merge commit: `988fa5abfb1aa344cfe526de5a5bb0629c2b6b36`
- Tiny model config: `configs/kgpt_tiny.yaml`
- Tiny tokenized-data config: `configs/kgpt_tiny_tokenized.yaml`
- Fixed-prompt eval config: `configs/eval_fixed_prompts.yaml`
- Data source manifest: `docs/phase04a_data_manifest.json`
- Completed report: `docs/phase04a_tiny_report.md`
- Pretraining implementation: `kgpt/pretrain.py`, `train/pretrain.py`
- Report implementation: `eval/report.py`
- Ignored tiny run: `experiments/runs/phase04a_tiny_smoke/`
- Validation commands: `uv run pytest`, `uv run ruff check .`, `git diff --check`, `uv run python -m train.pretrain --config configs/kgpt_tiny.yaml --max-steps 200 --run-name phase04a_tiny_smoke`, `uv run python -m eval.report --config configs/eval_fixed_prompts.yaml --checkpoint experiments/runs/phase04a_tiny_smoke/checkpoint_last.pt --output docs/phase04a_tiny_report.md`
- Resume probe: `uv run python -m train.pretrain --config configs/kgpt_tiny.yaml --max-steps 8 --run-name phase04a_runtime_probe --resume`
- Automation phase state was manually advanced after the live runner stalled in planning with no planner output.

### PHASE-03A - Core Decoder-Only Transformer

Status: complete
Completed: 2026-06-01
Evidence:
- Accepted automation plan: `runs/phase-runner/PHASE-03A/2026-05-31T17:12:39Z/accepted-plan/accepted-plan.json`
- PR: #3
- Merge commit: `d672096c7ebdbee1e09bf85431380e5105a4b9bc`
- Transformer config: `configs/transformer_micro.yaml`
- Architecture note: `docs/phase03a_transformer_architecture.md`
- Transformer implementation: `kgpt/transformer.py`
- Smoke trainer: `train/transformer_smoke.py`
- Generation CLI: `inference/generate.py`
- Ignored micro Transformer run: `experiments/runs/phase03a_transformer_micro/`
- Validation commands: `uv run pytest`, `uv run ruff check .`, `git diff --check`, `uv run python -m train.transformer_smoke --config configs/transformer_micro.yaml --max-steps 20`, `uv run python -m inference.generate --config configs/transformer_micro.yaml --prompt hello --max-new-tokens 16 --seed 123`
- Automation phase state was manually advanced after the nested executor stalled during the original live run.

### PHASE-02A - Tokenizer And Dataset Pipeline

Status: complete
Completed: 2026-06-01
Evidence:
- Accepted automation plan: `runs/phase-runner/PHASE-02A/2026-05-31T16:41:04Z/accepted-plan/accepted-plan.json`
- PR: #2
- Merge commit: `a1295e568294980f11dff5d7590ce5890d1b9d24`
- Tokenizer config: `configs/tokenizer_bilingual.yaml`
- Tokenized smoke config: `configs/tokenized_smoke.yaml`
- Tokenizer report: `docs/tokenizer_report.md`
- Data source manifest: `docs/phase02a_data_manifest.json`
- Ignored tokenizer model: `data/tokenized/tokenizers/phase02a-byte-bpe-bilingual.json`
- Ignored tokenized split metadata and token files: `data/tokenized/phase02a_smoke/`
- Validation commands: `uv run pytest`, `uv run ruff check .`, `git diff --check`, `uv run python -m tokenizer.train_report --config configs/tokenizer_bilingual.yaml --output docs/tokenizer_report.md`, `uv run python -m train.sample_batches --config configs/tokenized_smoke.yaml --max-batches 2`
- Automation phase state was manually advanced after the runner's premature remote-check final-gate block.

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
