import { execFile } from 'node:child_process';
import { access, readFile } from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';

import { loadAutopilotConfig, type AutopilotConfig } from './phase-autopilot.js';
import {
  buildPhaseRunBundle,
  defaultRunnerPaths,
  getRunnablePhases,
  loadPhaseRunnerConfig,
  validatePhaseGraph,
  type PhaseDefinition,
  type RunnerPaths,
} from './phase-runner.js';

export type ReadinessTarget = 'phase00b-auto' | 'unattended';
export type ReadinessStatus = 'pass' | 'warn' | 'fail';

export interface ReadinessCheck {
  id: string;
  status: ReadinessStatus;
  message: string;
  details?: unknown;
}

export interface ReadinessReport {
  schemaVersion: 1;
  target: ReadinessTarget;
  status: ReadinessStatus;
  score: string;
  repoRoot: string;
  checks: ReadinessCheck[];
  recommendedNextActions: string[];
}

export interface ReadinessCommandResult {
  exitCode: number;
  stdout: string;
  stderr: string;
}

export type ReadinessCommandRunner = (
  command: string,
  args: string[],
  options: { cwd: string },
) => Promise<ReadinessCommandResult>;

export interface ReadinessOptions {
  target: ReadinessTarget;
  commandRunner?: ReadinessCommandRunner;
  runnerPaths?: RunnerPaths;
  autopilotConfigPath?: string;
}

const execFileAsync = promisify(execFile);

const defaultCommandRunner: ReadinessCommandRunner = async (command, args, options) => {
  try {
    const result = await execFileAsync(command, args, {
      cwd: options.cwd,
      encoding: 'utf8',
      timeout: 30000,
    });
    return { exitCode: 0, stdout: result.stdout, stderr: result.stderr };
  } catch (error) {
    const commandError = error as {
      code?: number | string;
      stdout?: string;
      stderr?: string;
      message?: string;
    };
    return {
      exitCode: typeof commandError.code === 'number' ? commandError.code : 1,
      stdout: commandError.stdout ?? '',
      stderr: commandError.stderr ?? commandError.message ?? '',
    };
  }
};

const exists = async (filePath: string): Promise<boolean> =>
  access(filePath)
    .then(() => true)
    .catch(() => false);

const statusFromChecks = (checks: ReadinessCheck[]): ReadinessStatus => {
  if (checks.some((check) => check.status === 'fail')) return 'fail';
  if (checks.some((check) => check.status === 'warn')) return 'warn';
  return 'pass';
};

const scoreFromChecks = (checks: ReadinessCheck[]): string => {
  const failures = checks.filter((check) => check.status === 'fail').length;
  if (failures === 0 && checks.every((check) => check.status === 'pass')) {
    return '10/10';
  }
  return `${Math.max(0, 10 - failures)}/10`;
};

const relevantPhases = (target: ReadinessTarget, phases: PhaseDefinition[]): PhaseDefinition[] =>
  target === 'phase00b-auto'
    ? phases.filter((phase) => phase.id === 'PHASE-00B')
    : phases;

const parseGitClean = (stdout: string): boolean =>
  stdout
    .split('\n')
    .filter((line) => line.trim().length > 0)
    .every((line) => line.startsWith('## '));

const agentProvidersReady = (config: AutopilotConfig): boolean =>
  ['planner', 'executor', 'rechecker'].every(
    (role) => config.agents[role as 'planner' | 'executor' | 'rechecker']?.provider === 'shell',
  );

export const runReadiness = async (
  repoRootInput: string,
  options: ReadinessOptions,
): Promise<ReadinessReport> => {
  const repoRoot = path.resolve(repoRootInput);
  const commandRunner = options.commandRunner ?? defaultCommandRunner;
  const runnerPaths = options.runnerPaths ?? defaultRunnerPaths(repoRoot);
  const autopilotConfigPath = options.autopilotConfigPath ?? path.join(repoRoot, 'automation', 'autopilot-config.json');
  const config = await loadPhaseRunnerConfig(repoRoot, runnerPaths);
  const autopilotConfig = await loadAutopilotConfig(repoRoot, autopilotConfigPath);
  const phases = relevantPhases(options.target, config.graph.phases);
  const checks: ReadinessCheck[] = [];

  const graphErrors = validatePhaseGraph(config.graph);
  checks.push(
    graphErrors.length === 0
      ? { id: 'phase-graph-valid', status: 'pass', message: 'Phase graph is valid.' }
      : {
          id: 'phase-graph-valid',
          status: 'fail',
          message: 'Phase graph has validation errors.',
          details: { graphErrors },
        },
  );

  const gitStatus = await commandRunner('git', ['status', '--short', '--branch'], { cwd: repoRoot });
  checks.push(
    gitStatus.exitCode === 0 && parseGitClean(gitStatus.stdout)
      ? { id: 'git-worktree-clean', status: 'pass', message: 'Repository worktree is clean.' }
      : {
          id: 'git-worktree-clean',
          status: 'fail',
          message: 'Repository must be clean before unattended automation.',
          details: { stdout: gitStatus.stdout, stderr: gitStatus.stderr },
        },
  );

  checks.push(
    phases.every((phase) => phase.automerge === true)
      ? { id: 'phase-automerge-enabled', status: 'pass', message: 'Target phases allow automerge.' }
      : {
          id: 'phase-automerge-enabled',
          status: 'fail',
          message: 'Every target phase must set automerge true for auto mode.',
          details: {
            phases: phases
              .filter((phase) => phase.automerge !== true)
              .map((phase) => phase.id),
          },
        },
  );

  checks.push(
    config.automergePolicy.enabled === true && config.automergePolicy.automationSafetyReviewed === true
      ? { id: 'automerge-policy-enabled', status: 'pass', message: 'Automerge policy is enabled and reviewed.' }
      : {
          id: 'automerge-policy-enabled',
          status: 'fail',
          message: 'Automerge policy must be enabled and marked reviewed.',
        },
  );

  checks.push(
    config.automergePolicy.remoteChecks?.mode === 'hybrid'
      ? { id: 'remote-gate-hybrid', status: 'pass', message: 'Hybrid remote check policy is configured.' }
      : {
          id: 'remote-gate-hybrid',
          status: 'fail',
          message: 'Hybrid remote check policy is required for this readiness target.',
          details: { remoteChecks: config.automergePolicy.remoteChecks },
        },
  );

  checks.push(
    phases.every((phase) => (phase.validationCommands ?? []).length > 0)
      ? { id: 'phase-validation-commands', status: 'pass', message: 'Target phases define phase-specific validation commands.' }
      : {
          id: 'phase-validation-commands',
          status: 'fail',
          message: 'Every target phase must define phase-specific validation commands.',
          details: {
            phases: phases
              .filter((phase) => (phase.validationCommands ?? []).length === 0)
              .map((phase) => phase.id),
          },
        },
  );

  checks.push(
    await exists(path.join(repoRoot, 'automation', 'policies', 'unattended-decisions.json'))
      ? { id: 'unattended-decision-policy', status: 'pass', message: 'Unattended decision policy exists.' }
      : {
          id: 'unattended-decision-policy',
          status: 'fail',
          message: 'Unattended decision policy is missing.',
        },
  );

  checks.push(
    await exists(path.join(repoRoot, '.github', 'workflows', 'ci.yml'))
      ? { id: 'ci-workflow-present', status: 'pass', message: 'GitHub Actions CI workflow exists.' }
      : {
          id: 'ci-workflow-present',
          status: 'fail',
          message: 'GitHub Actions CI workflow is missing.',
        },
  );

  const ghAuth = await commandRunner('gh', ['auth', 'status'], { cwd: repoRoot });
  checks.push(
    ghAuth.exitCode === 0
      ? { id: 'github-auth-valid', status: 'pass', message: 'GitHub CLI auth is valid.' }
      : {
          id: 'github-auth-valid',
          status: 'fail',
          message: 'GitHub CLI auth must be valid for auto PR and merge.',
          details: { stderr: ghAuth.stderr, stdout: ghAuth.stdout },
        },
  );

  const remote = await commandRunner('git', ['ls-remote', '--heads', 'origin', autopilotConfig.git.baseBranch], {
    cwd: repoRoot,
  });
  checks.push(
    remote.exitCode === 0 && remote.stdout.trim().length > 0
      ? { id: 'remote-reachable', status: 'pass', message: 'Git remote is reachable.' }
      : {
          id: 'remote-reachable',
          status: 'fail',
          message: 'Git remote must be reachable for unattended automation.',
          details: { stdout: remote.stdout, stderr: remote.stderr },
        },
  );

  checks.push(
    agentProvidersReady(autopilotConfig)
      ? { id: 'shell-agents-configured', status: 'pass', message: 'Planner, executor, and rechecker use shell agents.' }
      : {
          id: 'shell-agents-configured',
          status: 'fail',
          message: 'Planner, executor, and rechecker must use shell agents for unattended automation.',
          details: { agents: autopilotConfig.agents },
        },
  );

  const runnable = getRunnablePhases(config, {
    repoRoot,
    from: options.target === 'phase00b-auto' ? 'PHASE-00B' : config.state.currentPhase,
    parallel: 1,
  });
  checks.push(
    runnable.length > 0
      ? { id: 'next-phase-runnable', status: 'pass', message: 'At least one target phase is runnable.' }
      : {
          id: 'next-phase-runnable',
          status: 'fail',
          message: 'No target phase is currently runnable.',
        },
  );

  for (const phase of phases) {
    await buildPhaseRunBundle(config, repoRoot, phase.id, 'readiness-sim', runnerPaths, {
      preflightCommands: autopilotConfig.preflightCommands,
    });
  }
  checks.push({
    id: 'dry-run-bundle-simulation',
    status: 'pass',
    message: 'Phase bundles can be built for target phases.',
  });

  const status = statusFromChecks(checks);
  return {
    schemaVersion: 1,
    target: options.target,
    status,
    score: scoreFromChecks(checks),
    repoRoot,
    checks,
    recommendedNextActions:
      status === 'pass'
        ? ['Run the target auto command.']
        : checks
            .filter((check) => check.status === 'fail')
            .map((check) => `Fix readiness check: ${check.id}`),
  };
};
