from __future__ import annotations

from types import SimpleNamespace

from inference import chat
from inference.runtime import load_inference_config, truncate_at_stop_string


def test_inference_config_loads_smoke_defaults() -> None:
    config = load_inference_config("configs/inference_smoke.yaml")

    assert config.model_kind == "transformer_smoke"
    assert config.bootstrap_if_missing
    assert config.generation["max_new_tokens"] == 16
    assert config.chat_template == "Q:{instruction}\nA:"


def test_inference_config_loads_phase11a_pretrain_checkpoint() -> None:
    config = load_inference_config("configs/inference_corpus_v01.yaml")

    assert config.model_kind == "pretrain"
    assert config.model_config.name == "kgpt_30m_corpus_v01.yaml"
    assert config.checkpoint.name == "checkpoint_last.pt"
    assert not config.bootstrap_if_missing
    assert config.generation["max_new_tokens"] == 16


def test_stop_string_truncates_decoded_completion() -> None:
    text, reason = truncate_at_stop_string(
        generated_text="hello world<stop>tail",
        prompt="hello ",
        stop_strings=["<stop>"],
    )

    assert text == "hello world"
    assert reason == "stop_string:<stop>"


def test_chat_cli_exposes_completion_generation_controls(monkeypatch) -> None:
    captured = {}
    config = SimpleNamespace(
        generation={
            "max_new_tokens": 8,
            "temperature": 0.0,
            "top_k": None,
            "top_p": None,
            "repetition_penalty": 1.0,
            "use_cache": True,
            "stop_strings": ["<eos>"],
            "stop_token_ids": [2],
        },
        chat_template="Q:{instruction}\nA:",
        dtype="float32",
    )

    monkeypatch.setattr(chat, "load_inference_config", lambda path: config)
    monkeypatch.setattr(chat, "load_model_for_inference", lambda **kwargs: object())

    def fake_generate_completion(**kwargs):
        captured.update(kwargs)
        return {"generated_text": "ok"}

    monkeypatch.setattr(chat, "generate_completion", fake_generate_completion)
    monkeypatch.setattr(chat, "write_json_or_print", lambda payload, output_path: captured.update(payload=payload))

    chat.main(
        [
            "--config",
            "configs/inference_smoke.yaml",
            "--instruction",
            "say hi",
            "--repetition-penalty",
            "1.2",
            "--stop-string",
            "<stop>",
            "--stop-token-id",
            "9",
            "--no-cache",
        ]
    )

    assert captured["prompt"] == "Q:say hi\nA:"
    assert captured["repetition_penalty"] == 1.2
    assert captured["stop_strings"] == ["<eos>", "<stop>"]
    assert captured["stop_token_ids"] == {2, 9}
    assert captured["use_cache"] is False
    assert captured["payload"]["dtype"] == "float32"
