import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { runInitCommand } from '../src/cli/commands/init.js';
import { runReadiness, type ReadinessCommandRunner } from '../src/core/readiness.js';

const passingCommandRunner: ReadinessCommandRunner = async (command, args) => {
  const joined = [command, ...args].join(' ');
  if (joined === 'git status --short --branch') {
    return { exitCode: 0, stdout: '## main...origin/main\n', stderr: '' };
  }
  if (joined === 'gh auth status') {
    return { exitCode: 0, stdout: 'Logged in to github.com\n', stderr: '' };
  }
  if (joined.startsWith('git ls-remote --heads origin')) {
    return { exitCode: 0, stdout: 'abc123\trefs/heads/main\n', stderr: '' };
  }
  return { exitCode: 0, stdout: '', stderr: '' };
};

const prepareReadyRepo = async (repoRoot: string): Promise<void> => {
  await runInitCommand(repoRoot, {});
  await mkdir(path.join(repoRoot, '.github', 'workflows'), { recursive: true });
  await writeFile(path.join(repoRoot, '.github', 'workflows', 'ci.yml'), 'name: CI\n');
  await mkdir(path.join(repoRoot, 'automation', 'policies'), { recursive: true });
  await writeFile(
    path.join(repoRoot, 'automation', 'policies', 'unattended-decisions.json'),
    JSON.stringify({ schemaVersion: 1, mode: 'agent-decides-within-policy' }, null, 2),
  );
  await writeFile(
    path.join(repoRoot, 'automation', 'phase-graph.json'),
    JSON.stringify({
      schemaVersion: 1,
      defaultStartPhase: 'PHASE-00B',
      defaultParallelism: 1,
      globalValidationCommands: ['git diff --check'],
      phases: [
        {
          id: 'PHASE-00B',
          plan: 'phase-plans/PHASE-TEMPLATE.md',
          dependsOn: [],
          allowedPaths: ['README.md', 'PROGRESS.md'],
          parallelGroup: 'foundation',
          automerge: true,
          validationCommands: [
            {
              id: 'phase-00b-smoke',
              command: 'git diff --check',
              requiredFor: ['phase', 'merge'],
            },
          ],
        },
      ],
    }, null, 2),
  );
  await writeFile(
    path.join(repoRoot, 'automation', 'phase-state.json'),
    JSON.stringify({
      schemaVersion: 1,
      lastUpdated: '2026-05-31',
      currentPhase: 'PHASE-00B',
      phases: { 'PHASE-00B': { status: 'queued' } },
    }, null, 2),
  );
  await writeFile(
    path.join(repoRoot, 'automation', 'policies', 'automerge-policy.json'),
    JSON.stringify({
      schemaVersion: 1,
      enabled: true,
      automationSafetyReviewed: true,
      mergeMethod: 'squash',
      deleteBranchAfterMerge: true,
      removeCleanWorktreeAfterMerge: true,
      allowNoRemoteChecksWhenLocalGatePasses: false,
      remoteChecks: { mode: 'hybrid', localOnlyPhases: ['PHASE-00B'] },
      requiredLocalCommands: ['git diff --check'],
      requiredPreflight: ['git status --short --branch'],
      requiredArtifacts: [],
      blockMergeWhen: [],
      gapPolicy: { blocking: 'block', non_blocking: 'allow', out_of_scope: 'allow' },
    }, null, 2),
  );
  const autopilotPath = path.join(repoRoot, 'automation', 'autopilot-config.json');
  const autopilot = JSON.parse(await readFile(autopilotPath, 'utf8')) as {
    agents: Record<string, { provider: string }>;
  };
  autopilot.agents.planner.provider = 'shell';
  autopilot.agents.executor.provider = 'shell';
  autopilot.agents.rechecker.provider = 'shell';
  await writeFile(autopilotPath, JSON.stringify(autopilot, null, 2));
};

describe('readiness', () => {
  it('passes when auto-mode prerequisites are present', async () => {
    const repoRoot = await mkdtemp(path.join(os.tmpdir(), 'agentic-readiness-test-'));
    try {
      await prepareReadyRepo(repoRoot);
      const report = await runReadiness(repoRoot, {
        target: 'phase00b-auto',
        commandRunner: passingCommandRunner,
      });

      expect(report.status).toBe('pass');
      expect(report.score).toBe('10/10');
    } finally {
      await rm(repoRoot, { recursive: true, force: true });
    }
  });

  it('fails when GitHub auth is unavailable', async () => {
    const repoRoot = await mkdtemp(path.join(os.tmpdir(), 'agentic-readiness-auth-test-'));
    try {
      await prepareReadyRepo(repoRoot);
      const report = await runReadiness(repoRoot, {
        target: 'phase00b-auto',
        commandRunner: async (command, args, options) => {
          if ([command, ...args].join(' ') === 'gh auth status') {
            return { exitCode: 1, stdout: '', stderr: 'not logged in' };
          }
          return passingCommandRunner(command, args, options);
        },
      });

      expect(report.status).toBe('fail');
      expect(report.checks.find((check) => check.id === 'github-auth-valid')?.status).toBe('fail');
    } finally {
      await rm(repoRoot, { recursive: true, force: true });
    }
  });
});
