from __future__ import annotations

import argparse
import json
from pathlib import Path

from kgpt.seed import seed_everything
from kgpt.transformer import (
    TransformerExperimentConfig,
    generate_tokens,
    load_tokenizer_for_config,
    load_transformer_checkpoint,
    load_transformer_experiment_config,
    resolve_device,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate text from a PHASE-03A decoder-only Transformer checkpoint.")
    parser.add_argument("--config", required=True, help="Path to transformer YAML config.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--checkpoint", help="Path to checkpoint_last.pt. Defaults to the configured smoke run.")
    parser.add_argument("--seed", type=int, default=123, help="Sampling seed.")
    parser.add_argument("--max-new-tokens", type=int, help="Override generation.max_new_tokens.")
    parser.add_argument("--temperature", type=float, help="Override generation.temperature; 0 means greedy decoding.")
    parser.add_argument("--top-k", type=int, help="Override generation.top_k; 0 disables top-k filtering.")
    parser.add_argument("--device", choices=["cpu", "mps"], help="Override configured generation device.")
    args = parser.parse_args(argv)

    config = load_transformer_experiment_config(args.config)
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else config.sampling.max_new_tokens
    temperature = args.temperature if args.temperature is not None else config.sampling.temperature
    top_k = args.top_k if args.top_k is not None else config.sampling.top_k
    if max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative.")
    if temperature < 0:
        raise ValueError("--temperature must be non-negative.")
    if top_k == 0:
        top_k = None
    if top_k is not None and top_k < 0:
        raise ValueError("--top-k must be non-negative.")

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else _default_checkpoint_path(config)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run train.transformer_smoke before generation."
        )

    seed_everything(args.seed)
    tokenizer = load_tokenizer_for_config(config)
    model, metadata = load_transformer_checkpoint(checkpoint_path, config=config, map_location="cpu")
    device = resolve_device(args.device or config.device)
    model.to(device)
    prompt_tokens = tokenizer.encode(args.prompt, add_bos=True)
    generated_token_ids = generate_tokens(
        model=model,
        input_ids=prompt_tokens,
        max_new_tokens=max_new_tokens,
        seed=args.seed,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id,
        device=device,
    )
    generated_text = tokenizer.decode(generated_token_ids)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "prompt": args.prompt,
                "generated_text": generated_text,
                "new_text": generated_text[len(args.prompt) :],
                "seed": args.seed,
                "temperature": temperature,
                "top_k": top_k,
                "max_new_tokens": max_new_tokens,
                "prompt_token_count": len(prompt_tokens),
                "generated_token_count": len(generated_token_ids),
                "device": str(device),
                "model_name": metadata.get("model_name"),
                "step": metadata.get("step"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _default_checkpoint_path(config: TransformerExperimentConfig) -> Path:
    return config.training.output_dir / config.run_name / "checkpoint_last.pt"


if __name__ == "__main__":
    raise SystemExit(main())
