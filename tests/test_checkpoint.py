import torch

from kgpt.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    path = tmp_path / "checkpoint.pt"
    metadata = {
        "config_hash": "abc",
        "config_path": "config.yaml",
        "model_name": "test-model",
        "step": 3,
        "seed": 7,
        "git_commit": "deadbeef",
        "created_at": "2026-05-31T00:00:00+00:00",
        "metrics": {"train_loss": 1.0},
        "tokenizer_id": "fake-tokenizer-v0",
    }

    save_checkpoint(path, model=model, optimizer=optimizer, metadata=metadata)
    restored = load_checkpoint(path, model=torch.nn.Linear(2, 2), optimizer=None)

    assert restored["metadata"]["schema_version"] == 1
    assert restored["metadata"]["model_name"] == "test-model"
    assert restored["metadata"]["step"] == 3
