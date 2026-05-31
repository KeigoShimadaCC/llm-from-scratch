from pathlib import Path


def test_generated_artifact_paths_are_ignored() -> None:
    gitignore = Path(".gitignore").read_text()

    assert "experiments/runs/" in gitignore
    assert "data/raw/" in gitignore
    assert "*.pt" in gitignore
    assert "*.safetensors" in gitignore
