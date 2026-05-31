import json
from pathlib import Path

import yaml

from kgpt.micro_char import generate_text, load_micro_char_checkpoint
from train.micro_char import main


def test_micro_char_training_overfits_and_generation_is_deterministic(tmp_path) -> None:
    config_path = tmp_path / "micro_char.yaml"
    config = yaml.safe_load(Path("configs/micro_char.yaml").read_text(encoding="utf8"))
    config["output_dir"] = str(tmp_path / "runs")
    config["train_steps"] = 120
    config["eval_every"] = 20
    config["sample_every"] = 60
    config_path.write_text(yaml.safe_dump(config), encoding="utf8")

    assert main(["--config", str(config_path), "--run-name", "pytest_micro_char"]) == 0

    run_dir = tmp_path / "runs" / "pytest_micro_char"
    checkpoint_path = run_dir / "checkpoint_last.pt"
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "samples.txt").is_file()
    assert (run_dir / "tokenizer.json").is_file()
    assert checkpoint_path.is_file()
    assert (run_dir / "manifest.json").is_file()

    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf8").splitlines()]
    assert metrics[0]["step"] == 0
    assert metrics[-1]["train_loss"] < metrics[0]["train_loss"]
    assert metrics[-1]["train_loss"] < config["overfit_threshold"]

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf8"))
    assert manifest["overfit_passed"] is True
    assert manifest["output_files"]["checkpoint"] == str(checkpoint_path)

    model, tokenizer, _metadata = load_micro_char_checkpoint(checkpoint_path)
    greedy_a = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        max_new_tokens=16,
        seed=1,
        temperature=0.0,
        greedy=True,
        device=next(model.parameters()).device,
    )
    greedy_b = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        max_new_tokens=16,
        seed=999,
        temperature=0.0,
        greedy=True,
        device=next(model.parameters()).device,
    )
    sampled_a = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        max_new_tokens=16,
        seed=123,
        temperature=0.8,
        greedy=False,
        device=next(model.parameters()).device,
    )
    sampled_b = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        max_new_tokens=16,
        seed=123,
        temperature=0.8,
        greedy=False,
        device=next(model.parameters()).device,
    )

    assert greedy_a == greedy_b
    assert sampled_a == sampled_b
    assert len(sampled_a) > len("hello")
