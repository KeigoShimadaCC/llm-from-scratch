from __future__ import annotations

from pathlib import Path

import torch

from kgpt.checkpoint import save_checkpoint
from kgpt.transformer import (
    DecoderOnlyTransformer,
    TransformerExperimentConfig,
    TransformerModelConfig,
    causal_mask,
    count_parameters,
    generate_tokens,
    load_transformer_checkpoint,
    resolve_device,
    shift_tokens_for_next_token_loss,
    shifted_cross_entropy,
)


def test_logits_shape_and_tied_embeddings() -> None:
    config = _model_config()
    model = DecoderOnlyTransformer(config)
    inputs = torch.randint(0, config.vocab_size, (2, 5))
    output = model(inputs)
    assert list(output.logits.shape) == [2, 5, config.vocab_size]
    assert model.lm_head.weight is model.token_embedding.weight
    assert count_parameters(model) > 0


def test_causal_mask_blocks_future_positions() -> None:
    mask = causal_mask(4)
    assert mask.dtype is torch.bool
    assert bool(mask[0, 0, 2, 1])
    assert not bool(mask[0, 0, 1, 2])


def test_future_token_change_cannot_change_earlier_logits() -> None:
    torch.manual_seed(7)
    model = DecoderOnlyTransformer(_model_config(dropout=0.0))
    model.eval()
    base = torch.tensor([[1, 2, 3, 4, 5]])
    changed_future = torch.tensor([[1, 2, 3, 9, 9]])
    with torch.no_grad():
        base_logits = model(base).logits
        changed_logits = model(changed_future).logits
    torch.testing.assert_close(base_logits[:, :3], changed_logits[:, :3])


def test_loss_uses_explicit_next_token_shift() -> None:
    token_ids = torch.tensor([[1, 2, 3, 4]])
    inputs, targets = shift_tokens_for_next_token_loss(token_ids)
    assert inputs.tolist() == [[1, 2, 3]]
    assert targets.tolist() == [[2, 3, 4]]

    model = DecoderOnlyTransformer(_model_config())
    logits = model(inputs).logits
    loss = shifted_cross_entropy(logits, targets)
    assert loss.ndim == 0


def test_generation_is_seeded_and_appends_requested_tokens() -> None:
    torch.manual_seed(11)
    config = _model_config(context_length=4)
    model = DecoderOnlyTransformer(config)
    prompt = [1, 2]
    first = generate_tokens(
        model=model,
        input_ids=prompt,
        max_new_tokens=3,
        seed=123,
        temperature=0.8,
    )
    second = generate_tokens(
        model=model,
        input_ids=prompt,
        max_new_tokens=3,
        seed=123,
        temperature=0.8,
    )
    assert first == second
    assert first[: len(prompt)] == prompt
    assert len(first) == len(prompt) + 3


def test_mps_request_falls_back_to_cpu_when_unavailable() -> None:
    assert resolve_device("mps", mps_available=False).type == "cpu"


def test_transformer_checkpoint_roundtrip(tmp_path: Path) -> None:
    model_config = _model_config()
    model = DecoderOnlyTransformer(model_config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    experiment_config = _experiment_config(tmp_path, model_config)
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        metadata={"model_name": "kgpt-test", "step": 0},
    )
    loaded_model, metadata = load_transformer_checkpoint(checkpoint_path, config=experiment_config)
    assert metadata["model_name"] == "kgpt-test"
    for original, loaded in zip(model.parameters(), loaded_model.parameters(), strict=True):
        torch.testing.assert_close(original, loaded)


def _model_config(
    *,
    vocab_size: int = 17,
    context_length: int = 8,
    embedding_dim: int = 12,
    num_layers: int = 2,
    num_heads: int = 3,
    mlp_hidden_dim: int = 24,
    dropout: float = 0.0,
) -> TransformerModelConfig:
    return TransformerModelConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_hidden_dim=mlp_hidden_dim,
        dropout=dropout,
        tie_embeddings=True,
    )


def _experiment_config(tmp_path: Path, model_config: TransformerModelConfig) -> TransformerExperimentConfig:
    from kgpt.config import OptimizerConfig
    from kgpt.transformer import (
        TransformerDataConfig,
        TransformerSamplingConfig,
        TransformerTokenizerConfig,
        TransformerTrainingConfig,
    )

    return TransformerExperimentConfig(
        run_name="test",
        seed=123,
        device="cpu",
        dtype="float32",
        model_name="kgpt-test",
        optimizer=OptimizerConfig(name="adamw", learning_rate=0.001),
        model=model_config,
        training=TransformerTrainingConfig(
            batch_size=2,
            train_steps=1,
            output_dir=tmp_path,
            checkpoint_every=1,
            eval_every=1,
        ),
        sampling=TransformerSamplingConfig(max_new_tokens=2, temperature=0.0, top_k=None),
        data=TransformerDataConfig(
            tokenized_config=tmp_path / "tokenized.yaml",
            metadata_path=tmp_path / "metadata.json",
            split="train",
        ),
        tokenizer=TransformerTokenizerConfig(
            model_path=tmp_path / "tokenizer.json",
            fallback_training_config=None,
        ),
        source_path=tmp_path / "config.yaml",
        config_hash="test",
    )
