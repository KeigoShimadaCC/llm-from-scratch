from __future__ import annotations

import argparse
from pathlib import Path

from inference.runtime import (
    config_generation_value,
    generate_completion,
    load_inference_config,
    load_model_for_inference,
    normalize_stop_token_ids,
    normalize_top_k,
    write_json_or_print,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run compact instruction/chat-style generation.")
    parser.add_argument("--config", required=True, help="Path to inference YAML config.")
    parser.add_argument("--instruction", required=True, help="Instruction text.")
    parser.add_argument("--checkpoint", help="Override checkpoint_last.pt path.")
    parser.add_argument("--tokenizer", help="Override tokenizer JSON path.")
    parser.add_argument("--seed", type=int, default=123, help="Sampling seed.")
    parser.add_argument("--max-new-tokens", type=int, help="Override generation.max_new_tokens.")
    parser.add_argument("--temperature", type=float, help="Override generation.temperature.")
    parser.add_argument("--top-k", type=int, help="Override generation.top_k; 0 disables top-k.")
    parser.add_argument("--top-p", type=float, help="Override generation.top_p; 0 disables top-p.")
    parser.add_argument("--device", choices=["cpu", "mps"], help="Override configured generation device.")
    parser.add_argument("--use-cache", action="store_true", help="Use the KV-cache inference path.")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args(argv)

    config = load_inference_config(args.config)
    loaded = load_model_for_inference(
        config=config,
        checkpoint_override=Path(args.checkpoint) if args.checkpoint else None,
        device_override=args.device,
        tokenizer_override=Path(args.tokenizer) if args.tokenizer else None,
    )
    top_p = args.top_p
    if top_p == 0:
        top_p = None
    prompt = config.chat_template.format(instruction=args.instruction)
    payload = generate_completion(
        loaded=loaded,
        prompt=prompt,
        seed=args.seed,
        max_new_tokens=int(config_generation_value(config=config, name="max_new_tokens", override=args.max_new_tokens)),
        temperature=float(config_generation_value(config=config, name="temperature", override=args.temperature)),
        top_k=normalize_top_k(config_generation_value(config=config, name="top_k", override=args.top_k)),
        top_p=config_generation_value(config=config, name="top_p", override=top_p),
        repetition_penalty=float(
            config_generation_value(config=config, name="repetition_penalty", override=None)
        ),
        stop_strings=list(config.generation["stop_strings"]),
        stop_token_ids=normalize_stop_token_ids(None, config),
        use_cache=args.use_cache or bool(config.generation["use_cache"]),
    )
    payload["instruction"] = args.instruction
    payload["chat_template"] = config.chat_template
    payload["tokenizer"] = args.tokenizer or "config"
    write_json_or_print(payload, Path(args.output) if args.output else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
