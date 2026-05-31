from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from inference.runtime import generate_completion, load_inference_config, load_model_for_inference, memory_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark local inference devices.")
    parser.add_argument("--config", required=True, help="Path to inference benchmark YAML config.")
    parser.add_argument("--max-new-tokens", type=int, help="Override generation token count.")
    parser.add_argument("--output", required=True, help="Markdown benchmark report path.")
    args = parser.parse_args(argv)
    result = run_benchmark(
        config_path=Path(args.config),
        max_new_tokens=args.max_new_tokens,
        output_path=Path(args.output),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def run_benchmark(*, config_path: Path, max_new_tokens: int | None, output_path: Path) -> dict[str, Any]:
    config = load_inference_config(config_path)
    rows = []
    for device in config.benchmark["devices"]:
        rows.append(_benchmark_device(config=config, device=str(device), max_new_tokens=max_new_tokens))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_report(config_path=config_path, config=config, rows=rows), encoding="utf8")
    return {
        "output": str(output_path),
        "devices": [row["device"] for row in rows],
        "available_devices": [row["device"] for row in rows if row["status"] == "measured"],
        "mlx_status": next((row["status"] for row in rows if row["device"] == "mlx"), "not_requested"),
    }


def _benchmark_device(*, config: Any, device: str, max_new_tokens: int | None) -> dict[str, Any]:
    if device == "mlx":
        if importlib.util.find_spec("mlx") is None:
            return {
                "device": "mlx",
                "status": "deferred",
                "reason": f"MLX is not installed; see {config.mlx['deferral_doc']}.",
            }
        return {
            "device": "mlx",
            "status": "deferred",
            "reason": f"MLX model-loading parity is deferred; see {config.mlx['deferral_doc']}.",
        }
    if device == "mps" and not torch.backends.mps.is_available():
        return {"device": "mps", "status": "unavailable", "reason": "torch.backends.mps.is_available() is false."}
    loaded = load_model_for_inference(config=config, device_override=device)
    token_count = max_new_tokens or int(config.generation["max_new_tokens"])
    for _ in range(int(config.benchmark["warmup_runs"])):
        generate_completion(
            loaded=loaded,
            prompt=str(config.benchmark["prompts"][0]),
            seed=123,
            max_new_tokens=min(4, token_count),
            temperature=0.0,
            top_k=None,
            top_p=None,
            repetition_penalty=1.0,
            stop_strings=[],
            stop_token_ids=set(),
            use_cache=False,
        )
    measurements = []
    for run_index in range(int(config.benchmark["measured_runs"])):
        for prompt in config.benchmark["prompts"]:
            uncached = generate_completion(
                loaded=loaded,
                prompt=str(prompt),
                seed=123 + run_index,
                max_new_tokens=token_count,
                temperature=0.0,
                top_k=None,
                top_p=None,
                repetition_penalty=1.0,
                stop_strings=[],
                stop_token_ids=set(),
                use_cache=False,
            )
            cached = generate_completion(
                loaded=loaded,
                prompt=str(prompt),
                seed=123 + run_index,
                max_new_tokens=token_count,
                temperature=0.0,
                top_k=None,
                top_p=None,
                repetition_penalty=1.0,
                stop_strings=[],
                stop_token_ids=set(),
                use_cache=True,
            )
            measurements.append({"mode": "uncached", **uncached})
            measurements.append({"mode": "cached", **cached})
    return {
        "device": device,
        "status": "measured",
        "prompt_count": len(config.benchmark["prompts"]),
        "measured_runs": int(config.benchmark["measured_runs"]),
        "max_new_tokens": token_count,
        "uncached_tokens_per_sec": _mean(
            row["tokens_per_sec"] for row in measurements if row["mode"] == "uncached"
        ),
        "cached_tokens_per_sec": _mean(
            row["tokens_per_sec"] for row in measurements if row["mode"] == "cached"
        ),
        "uncached_latency_sec": _mean(row["latency_sec"] for row in measurements if row["mode"] == "uncached"),
        "cached_latency_sec": _mean(row["latency_sec"] for row in measurements if row["mode"] == "cached"),
        "memory": memory_snapshot(loaded.device),
    }


def _render_report(*, config_path: Path, config: Any, rows: list[dict[str, Any]]) -> str:
    table = "\n".join(
        "| {device} | {status} | {uncached} | {cached} | {latency} | {memory} | {reason} |".format(
            device=row["device"],
            status=row["status"],
            uncached=_fmt(row.get("uncached_tokens_per_sec")),
            cached=_fmt(row.get("cached_tokens_per_sec")),
            latency=_fmt(row.get("cached_latency_sec")),
            memory=json.dumps(row.get("memory", {}), sort_keys=True),
            reason=row.get("reason", ""),
        )
        for row in rows
    )
    prompts = "\n".join(f"- `{prompt}`" for prompt in config.benchmark["prompts"])
    return f"""# PHASE-08A Inference Benchmark

## Protocol

- Config: `{config_path}`
- Checkpoint: `{config.checkpoint}`
- Model config: `{config.model_config}`
- Prompt count: {len(config.benchmark["prompts"])}
- Warmup runs: {config.benchmark["warmup_runs"]}
- Measured runs: {config.benchmark["measured_runs"]}
- Default max new tokens: {config.generation["max_new_tokens"]}
- Sampling: greedy for benchmark stability.

## Prompts

{prompts}

## Results

| Device | Status | Uncached tok/s | Cached tok/s | Cached latency sec | Memory | Reason |
|---|---:|---:|---:|---:|---|---|
{table}

## MLX Status

MLX is compared when an importable `mlx` runtime and a compatible scratch-model loading path are available. If the
table marks MLX as deferred, see `{config.mlx["deferral_doc"]}` for blocker evidence and follow-up.
"""


def _mean(values: Any) -> float:
    values_list = [float(value) for value in values]
    return statistics.fmean(values_list) if values_list else 0.0


def _fmt(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.4f}"
    return "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
