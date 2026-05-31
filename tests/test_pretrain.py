from __future__ import annotations

from pathlib import Path

import torch

from kgpt.checkpoint import save_checkpoint
from kgpt.pretrain import (
    SchedulerConfig,
    learning_rate_for_step,
    load_pretrain_config,
    load_resume_state,
)
from kgpt.transformer import DecoderOnlyTransformer, count_parameters


def test_tiny_config_is_in_phase_scale_range() -> None:
    config = load_pretrain_config("configs/kgpt_tiny.yaml")
    model = DecoderOnlyTransformer(config.model)
    parameter_count = count_parameters(model)
    assert config.scale.min_parameters <= parameter_count <= config.scale.max_parameters
    assert config.training.loss_improvement_threshold == 0.10


def test_kgpt_30m_config_crosses_north_star_gate() -> None:
    config = load_pretrain_config("configs/kgpt_30m.yaml")
    model = DecoderOnlyTransformer(config.model)
    parameter_count = count_parameters(model)
    assert parameter_count >= 30_000_000
    assert config.scale.min_parameters <= parameter_count <= config.scale.max_parameters


def test_learning_rate_schedule_warms_up_and_decays() -> None:
    scheduler = SchedulerConfig(warmup_steps=2, min_lr_factor=0.1)
    assert learning_rate_for_step(base_lr=1.0, step=0, total_steps=10, scheduler=scheduler) == 0.0
    assert learning_rate_for_step(base_lr=1.0, step=1, total_steps=10, scheduler=scheduler) == 0.5
    assert learning_rate_for_step(base_lr=1.0, step=2, total_steps=10, scheduler=scheduler) == 1.0
    assert learning_rate_for_step(base_lr=1.0, step=10, total_steps=10, scheduler=scheduler) == 0.1


def test_resume_state_loads_checkpoint_metadata(tmp_path: Path) -> None:
    config = load_pretrain_config("configs/kgpt_tiny.yaml")
    model = DecoderOnlyTransformer(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.learning_rate)
    save_checkpoint(
        tmp_path / "checkpoint_last.pt",
        model=model,
        optimizer=optimizer,
        metadata={
            "step": 12,
            "best_validation_loss": 3.5,
            "best_step": 10,
            "initial_validation_loss": 6.0,
        },
    )
    state = load_resume_state(run_dir=tmp_path, model=model, optimizer=optimizer)
    assert state.start_step == 12
    assert state.best_validation_loss == 3.5
    assert state.best_step == 10
    assert state.initial_validation_loss == 6.0
