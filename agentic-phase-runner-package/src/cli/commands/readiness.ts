import { runReadiness, type ReadinessTarget } from '../../core/readiness.js';
import { loadRunnerContext, optionValue, writeJson } from './shared.js';

const parseTarget = (value: string | undefined): ReadinessTarget => {
  if (value === 'phase00b-auto' || value === 'unattended') {
    return value;
  }
  throw new Error('--target must be one of: phase00b-auto, unattended');
};

export const runReadinessCommand = async (
  repoRoot: string,
  options: Record<string, string | boolean>,
): Promise<void> => {
  const { autopilotConfigPath, paths } = await loadRunnerContext(repoRoot);
  writeJson(
    await runReadiness(repoRoot, {
      target: parseTarget(optionValue(options, 'target')),
      runnerPaths: paths,
      autopilotConfigPath,
    }),
  );
};
