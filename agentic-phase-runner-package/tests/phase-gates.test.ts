import { describe, expect, it } from 'vitest';

import {
  evaluateAutomerge,
  evaluatePhaseAcceptanceGate,
  type AutomergePolicy,
  type PhaseDefinition,
  type PhaseMergeEvidence,
} from '../src/core/phase-runner.js';

const phase = {
  id: 'PHASE-00B',
  plan: 'phase-plans/PHASE-00B.md',
  dependsOn: [],
  allowedPaths: ['README.md', 'src/**'],
  parallelGroup: 'foundation',
  automerge: true,
} satisfies PhaseDefinition;

const evidence = {
  localCommands: [{ command: 'git diff --check', status: 'pass' }],
  remoteChecks: 'none',
  cursorRecheck: 'pass',
  phaseAcceptanceComplete: true,
  changedPaths: ['README.md', 'src/index.ts'],
  worktreeClean: true,
  secretsDetected: false,
  blockingGaps: [],
} satisfies PhaseMergeEvidence;

const policy = {
  schemaVersion: 1,
  enabled: false,
  automationSafetyReviewed: false,
  mergeMethod: 'squash',
  deleteBranchAfterMerge: false,
  removeCleanWorktreeAfterMerge: false,
  allowNoRemoteChecksWhenLocalGatePasses: false,
  remoteChecks: {
    mode: 'required',
    localOnlyPhases: [],
  },
  requiredLocalCommands: ['git diff --check'],
  requiredPreflight: ['git status --short --branch'],
  requiredArtifacts: [],
  blockMergeWhen: [],
  gapPolicy: {
    blocking: 'block',
    non_blocking: 'allow',
    out_of_scope: 'allow',
  },
} satisfies AutomergePolicy;

describe('phase gates', () => {
  it('allows local phase acceptance without requiring automerge policy', () => {
    expect(evaluatePhaseAcceptanceGate(phase, evidence).decision).toBe('allow');
    expect(evaluateAutomerge(phase, policy, evidence).decision).toBe('block');
  });

  it('allows hybrid local-only phases when remote checks are absent', () => {
    const decision = evaluateAutomerge(phase, {
      ...policy,
      enabled: true,
      automationSafetyReviewed: true,
      remoteChecks: {
        mode: 'hybrid',
        localOnlyPhases: ['PHASE-00B'],
      },
    }, evidence);

    expect(decision.decision).toBe('allow');
  });

  it('blocks changed paths outside the phase scope', () => {
    const decision = evaluatePhaseAcceptanceGate(phase, {
      ...evidence,
      changedPaths: ['.env'],
    });

    expect(decision.decision).toBe('block');
    expect(decision.reasons.join('\n')).toContain('Changed path is outside phase scope');
  });
});
