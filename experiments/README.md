# Experiment Artifacts

Training and evaluation runs should write under ignored run directories, usually:

```text
experiments/runs/{timestamp}_{run_name}/
```

Each serious run should preserve:

- `config.yaml` or equivalent config artifact;
- `metrics.jsonl`;
- generated samples;
- checkpoint metadata;
- tokenizer metadata when relevant;
- evaluation report.

Large checkpoints, generated run directories, and local benchmark outputs must not be committed.

PHASE-00B dummy training writes:

- `config.yaml`
- `metrics.jsonl`
- `samples.txt`
- `checkpoint_last.pt`
- `manifest.json`

Those files are generated evidence and must remain under ignored `experiments/runs/` directories.
