from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from kgpt.checkpoint import load_checkpoint
from kgpt.sft import build_sft_examples, load_base_pretrain_config, load_sft_config
from kgpt.transformer import DecoderOnlyTransformer, generate_tokens, load_tokenizer_for_config, resolve_device
from train.sft import evaluate_sft_loss


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare base and SFT checkpoints on fixed instruction probes.")
    parser.add_argument("--config", required=True, help="Path to SFT eval YAML config.")
    parser.add_argument("--output", required=True, help="Markdown report output path.")
    args = parser.parse_args(argv)
    result = compare_sft(eval_config_path=Path(args.config), output_path=Path(args.output))
    print(json.dumps(result, sort_keys=True))
    return 0


def compare_sft(*, eval_config_path: Path, output_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(eval_config_path.read_text(encoding="utf8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Eval config must be a mapping: {eval_config_path}")
    sft_config = load_sft_config(raw["sft_config"])
    base_config = load_base_pretrain_config(sft_config)
    tokenizer = load_tokenizer_for_config(base_config)
    records_by_pair = {
        (record.instruction, record.response): record
        for record in sft_config.dataset.records
    }
    probes = tuple(
        records_by_pair[(item["instruction"], item["expected_response"])]
        for item in raw["fixed_probes"]
    )
    examples = build_sft_examples(
        records=probes,
        tokenizer=tokenizer,
        template=sft_config.prompt_template,
        context_length=base_config.model.context_length,
        response_only_loss=sft_config.dataset.response_only_loss,
    )
    device = resolve_device(sft_config.device)
    base_model = _load_model(base_config.model, Path(raw["base_checkpoint"]), device=device)
    sft_model = _load_model(base_config.model, Path(raw["sft_checkpoint"]), device=device)
    base_loss = evaluate_sft_loss(model=base_model, examples=examples, device=device)
    sft_loss = evaluate_sft_loss(model=sft_model, examples=examples, device=device)
    sft_manifest = json.loads((Path(raw["sft_checkpoint"]).parent / "manifest.json").read_text(encoding="utf8"))
    samples = [
        {
            "instruction": probe.instruction,
            "expected": probe.response,
            "base": _generate(base_model, tokenizer, sft_config, probe.instruction, device=device),
            "sft": _generate(sft_model, tokenizer, sft_config, probe.instruction, device=device),
        }
        for probe in probes
    ]
    report_text = _render_report(
        title=str(raw.get("title", "PHASE-06A SFT comparison")),
        sft_config=sft_config,
        sft_manifest=sft_manifest,
        base_loss=base_loss,
        sft_loss=sft_loss,
        samples=samples,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf8")
    run_eval_report = _write_sft_run_eval_report(
        checkpoint_path=Path(raw["sft_checkpoint"]),
        report_text=report_text,
    )
    return {
        "output": str(output_path),
        "base_loss": base_loss,
        "sft_loss": sft_loss,
        "loss_improved": sft_loss < base_loss,
        "probe_count": len(samples),
        "run_eval_report": str(run_eval_report) if run_eval_report else None,
    }


def _load_model(model_config: Any, checkpoint_path: Path, *, device: torch.device) -> DecoderOnlyTransformer:
    model = DecoderOnlyTransformer(model_config)
    load_checkpoint(checkpoint_path, model=model, map_location="cpu")
    model.to(device)
    model.eval()
    return model


def _generate(
    model: DecoderOnlyTransformer,
    tokenizer: Any,
    sft_config: Any,
    instruction: str,
    *,
    device: torch.device,
) -> str:
    prompt = sft_config.prompt_template.prompt_pattern.format(instruction=instruction)
    token_ids = generate_tokens(
        model=model,
        input_ids=tokenizer.encode(prompt, add_bos=True),
        max_new_tokens=sft_config.sampling.max_new_tokens,
        seed=sft_config.seed,
        temperature=sft_config.sampling.temperature,
        top_k=sft_config.sampling.top_k,
        eos_token_id=tokenizer.eos_token_id,
        device=device,
    )
    return tokenizer.decode(token_ids)


def _write_sft_run_eval_report(*, checkpoint_path: Path, report_text: str) -> Path | None:
    manifest_path = checkpoint_path.parent / "manifest.json"
    if not manifest_path.is_file():
        return None
    eval_report_path = checkpoint_path.parent / "eval_report.md"
    eval_report_path.write_text(report_text, encoding="utf8")
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    output_files = manifest.setdefault("output_files", {})
    output_files["eval_report"] = str(eval_report_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf8")
    return eval_report_path


def _render_report(
    *,
    title: str,
    sft_config: Any,
    sft_manifest: dict[str, Any],
    base_loss: float,
    sft_loss: float,
    samples: list[dict[str, str]],
) -> str:
    sample_blocks = "\n".join(
        f"### `{sample['instruction']}`\n\n"
        f"- Expected: `{sample['expected']}`\n"
        f"- Base: `{sample['base']}`\n"
        f"- SFT: `{sample['sft']}`"
        for sample in samples
    )
    validation_line = (
        f'{sft_manifest["initial_validation_loss"]:.4f} -> {sft_manifest["final_validation_loss"]:.4f}'
    )
    return f"""# {title}

## Summary

- Prompt template version: `{sft_config.prompt_template.version}`
- Dataset source: {sft_config.dataset.source_name}
- License: {sft_config.dataset.license}
- Response-only loss masking: {sft_config.dataset.response_only_loss}
- Base response loss: {base_loss:.4f}
- SFT response loss: {sft_loss:.4f}
- Narrow improvement: {"yes" if sft_loss < base_loss else "no"}
- Held-out SFT validation loss: {validation_line}

## Prompt Format

```text
{sft_config.prompt_template.prompt_pattern}
```

## Fixed Probe Samples

{sample_blocks}

## Limitations

- This is a tiny repo-authored instruction fixture, not a safety or alignment dataset.
- The model is only expected to improve on narrow command/response probes.
- Held-out validation loss regressed in this smoke run, so the evidence supports memorized narrow commands only.
- Generated text can still repeat or include tokenization artifacts.
"""


if __name__ == "__main__":
    raise SystemExit(main())
