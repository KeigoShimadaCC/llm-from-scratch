from pathlib import Path

from kgpt.config import load_training_config


def test_load_dummy_config() -> None:
    config = load_training_config(Path("configs/dummy.yaml"))

    assert config.run_name == "dummy"
    assert config.vocab_size == 128
    assert config.context_length == 16
    assert config.optimizer.name == "adamw"
    assert len(config.config_hash) == 64
