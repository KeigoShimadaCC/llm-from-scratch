from __future__ import annotations

from kgpt.sft import (
    IGNORE_INDEX,
    build_sft_examples,
    format_prompt,
    load_sft_config,
    split_instruction_records,
)
from kgpt.transformer import load_tokenizer_for_config


def test_sft_prompt_template_version_and_format() -> None:
    config = load_sft_config("configs/sft_smoke.yaml")
    assert config.prompt_template.version == "kgpt_sft_v1_compact"
    assert format_prompt(config.prompt_template, "say hi") == "Q:say hi\nA:"


def test_response_only_loss_masks_prompt_tokens() -> None:
    config = load_sft_config("configs/sft_smoke.yaml")
    base_config = config.base.config
    from kgpt.pretrain import load_pretrain_config

    pretrain_config = load_pretrain_config(base_config)
    tokenizer = load_tokenizer_for_config(pretrain_config)
    record = config.dataset.records[0]
    example = build_sft_examples(
        records=(record,),
        tokenizer=tokenizer,
        template=config.prompt_template,
        context_length=pretrain_config.model.context_length,
        response_only_loss=True,
    )[0]
    assert IGNORE_INDEX in example.targets.tolist()
    assert any(target != IGNORE_INDEX for target in example.targets.tolist())


def test_instruction_split_is_deterministic() -> None:
    config = load_sft_config("configs/sft_smoke.yaml")
    first = split_instruction_records(
        config.dataset.records,
        validation_fraction=config.dataset.validation_fraction,
        seed=config.dataset.split_seed,
    )
    second = split_instruction_records(
        config.dataset.records,
        validation_fraction=config.dataset.validation_fraction,
        seed=config.dataset.split_seed,
    )
    assert first == second
    assert first["train"]
    assert first["validation"]
