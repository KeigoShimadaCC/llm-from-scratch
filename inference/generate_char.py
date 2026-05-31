from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kgpt.micro_char import generate_text, load_micro_char_checkpoint
from kgpt.seed import seed_everything


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate text from a PHASE-01A micro character checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_last.pt.")
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt text. Characters must exist in the checkpoint vocabulary.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Sampling seed.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Number of new characters to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature. Use 0 with --greedy.")
    parser.add_argument("--greedy", action="store_true", help="Use argmax decoding instead of sampling.")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu", help="Generation device.")
    args = parser.parse_args(argv)

    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative.")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative.")

    seed_everything(args.seed)
    device = _resolve_device(args.device)
    model, tokenizer, metadata = load_micro_char_checkpoint(Path(args.checkpoint), map_location="cpu")
    model.to(device)
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        temperature=args.temperature,
        greedy=args.greedy,
        device=device,
    )
    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "prompt": args.prompt,
                "generated_text": generated_text,
                "new_text": generated_text[len(args.prompt) :],
                "seed": args.seed,
                "temperature": 0.0 if args.greedy else args.temperature,
                "greedy": args.greedy,
                "model_name": metadata.get("model_name"),
                "step": metadata.get("step"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _resolve_device(requested: str) -> torch.device:
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    raise SystemExit(main())
