from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference.runtime import generate_completion, load_inference_config, load_model_for_inference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check cached versus uncached deterministic generation parity.")
    parser.add_argument("--config", required=True, help="Path to inference YAML config.")
    parser.add_argument("--prompt", default="hello", help="Parity prompt.")
    parser.add_argument("--max-new-tokens", type=int, help="Override generated tokens for parity.")
    parser.add_argument("--seed", type=int, default=123, help="Sampling seed.")
    parser.add_argument("--device", choices=["cpu", "mps"], help="Override configured device.")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args(argv)

    config = load_inference_config(args.config)
    loaded = load_model_for_inference(config=config, device_override=args.device)
    max_new_tokens = args.max_new_tokens or min(8, int(config.generation["max_new_tokens"]))
    common = {
        "loaded": loaded,
        "prompt": args.prompt,
        "seed": args.seed,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "top_k": None,
        "top_p": None,
        "repetition_penalty": 1.0,
        "stop_strings": [],
        "stop_token_ids": set(),
    }
    uncached = generate_completion(**common, use_cache=False)
    cached = generate_completion(**common, use_cache=True)
    passed = uncached["generated_text"] == cached["generated_text"]
    payload = {
        "passed": passed,
        "prompt": args.prompt,
        "max_new_tokens": max_new_tokens,
        "uncached_text": uncached["generated_text"],
        "cached_text": cached["generated_text"],
        "uncached_tokens_per_sec": uncached["tokens_per_sec"],
        "cached_tokens_per_sec": cached["tokens_per_sec"],
        "device": uncached["device"],
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf8")
    print(text, end="")
    if not passed:
        raise RuntimeError("KV-cache parity failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
