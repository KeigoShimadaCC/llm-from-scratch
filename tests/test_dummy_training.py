import json
from pathlib import Path

import yaml

from kgpt.checkpoint import load_checkpoint
from train.dummy import main


def test_dummy_training_writes_expected_artifacts(tmp_path) -> None:
    config_path = tmp_path / "dummy.yaml"
    config = yaml.safe_load(Path("configs/dummy.yaml").read_text())
    config["output_dir"] = str(tmp_path / "runs")
    config["train_steps"] = 2
    config_path.write_text(yaml.safe_dump(config), encoding="utf8")

    assert main(["--config", str(config_path), "--run-name", "pytest_smoke"]) == 0

    run_dirs = list((tmp_path / "runs").glob("*_pytest_smoke"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "samples.txt").is_file()
    assert (run_dir / "checkpoint_last.pt").is_file()
    assert (run_dir / "manifest.json").is_file()

    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert metrics[-1]["step"] == 2
    checkpoint = load_checkpoint(run_dir / "checkpoint_last.pt")
    assert checkpoint["metadata"]["tokenizer_id"] == "fake-tokenizer-v0"
