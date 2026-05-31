from __future__ import annotations

from inference.runtime import load_inference_config, truncate_at_stop_string


def test_inference_config_loads_smoke_defaults() -> None:
    config = load_inference_config("configs/inference_smoke.yaml")

    assert config.model_kind == "transformer_smoke"
    assert config.bootstrap_if_missing
    assert config.generation["max_new_tokens"] == 16
    assert config.chat_template == "Q:{instruction}\nA:"


def test_stop_string_truncates_decoded_completion() -> None:
    text, reason = truncate_at_stop_string(
        generated_text="hello world<stop>tail",
        prompt="hello ",
        stop_strings=["<stop>"],
    )

    assert text == "hello world"
    assert reason == "stop_string:<stop>"
