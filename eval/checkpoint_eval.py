from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from kgpt.checkpoint import load_checkpoint
from kgpt.pretrain import load_pretrain_config, perplexity
from kgpt.sft import InstructionRecord, build_sft_examples, load_base_pretrain_config, load_sft_config
from kgpt.token_data import TokenBatchSampler, build_tokenized_dataset_from_config, load_yaml_config, normalize_text
from kgpt.token_data import records_from_config as token_records_from_config
from kgpt.transformer import (
    DecoderOnlyTransformer,
    count_parameters,
    generate_tokens,
    load_tokenizer_for_config,
    load_transformer_experiment_config,
    resolve_device,
)
from train.pretrain import evaluate_loss
from train.sft import evaluate_sft_loss
from train.transformer_smoke import evaluate_batch_loss

DEFAULT_SUMMARY_STATUS = "summary_only_missing_ignored_checkpoint"


def load_eval_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Eval config must be a mapping: {config_path}")
    generation = raw.get("generation") if isinstance(raw.get("generation"), dict) else {}
    validation = raw.get("validation") if isinstance(raw.get("validation"), dict) else {}
    return {
        "schema_version": int(raw.get("schema_version", 1)),
        "title": str(raw.get("title", "Checkpoint evaluation report")),
        "checkpoint_manifest": str(raw.get("checkpoint_manifest", "docs/checkpoint_manifest.json")),
        "fixed_prompts": _normalize_fixed_prompts(raw.get("fixed_prompts")),
        "toy_tasks": _normalize_toy_tasks(raw.get("toy_tasks")),
        "memorization": raw.get("memorization_probes", {}),
        "generation": {
            "max_new_tokens": int(generation.get("max_new_tokens", 32)),
            "temperature": float(generation.get("temperature", 0.0)),
            "top_k": None if generation.get("top_k") in {None, 0} else int(generation["top_k"]),
            "distribution_top_k": int(generation.get("distribution_top_k", 5)),
        },
        "validation": {
            "batches": int(validation.get("batches", 2)),
            "device": str(validation.get("device", "cpu")),
            "summary_fallback": bool(validation.get("summary_fallback", True)),
        },
        "metric_definitions": raw.get("metric_definitions", {}),
    }


def load_checkpoint_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf8"))
    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError(f"Checkpoint manifest must contain checkpoints: {manifest_path}")
    for entry in checkpoints:
        if not isinstance(entry, dict):
            raise ValueError("Checkpoint entries must be mappings.")
        for key in ("id", "label", "kind", "config", "checkpoint", "summary"):
            if key not in entry:
                raise ValueError(f"Checkpoint entry is missing {key}: {entry}")
    return payload


def evaluate_checkpoint_manifest(
    *,
    eval_config: dict[str, Any],
    checkpoint_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        evaluate_checkpoint_entry(entry=entry, eval_config=eval_config)
        for entry in checkpoint_manifest["checkpoints"]
    ]


def evaluate_checkpoint_entry(*, entry: dict[str, Any], eval_config: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = Path(entry["checkpoint"])
    started = time.perf_counter()
    if checkpoint_path.is_file():
        result = _evaluate_live_checkpoint(entry=entry, eval_config=eval_config, checkpoint_path=checkpoint_path)
        result["evaluation_wall_time_sec"] = time.perf_counter() - started
        return result
    if not eval_config["validation"]["summary_fallback"]:
        raise FileNotFoundError(f"Checkpoint artifact is missing and summary_fallback is disabled: {checkpoint_path}")
    return _summary_only_result(entry=entry, eval_config=eval_config, checkpoint_path=checkpoint_path)


def render_eval_report(
    *,
    title: str,
    eval_config: dict[str, Any],
    checkpoint_manifest: dict[str, Any],
    results: list[dict[str, Any]],
) -> str:
    metric_lines = _render_metric_definitions(eval_config)
    prompt_lines = "\n".join(
        f"- `{prompt['id']}` ({prompt['category']}): {prompt['prompt']}" for prompt in eval_config["fixed_prompts"]
    )
    toy_lines = "\n".join(
        f"- `{task['id']}`: `{task['prompt']}` -> `{task['expected_response']}`"
        for task in eval_config["toy_tasks"]
    )
    checkpoint_blocks = "\n\n".join(_render_checkpoint_result(result) for result in results)
    live_count = sum(1 for result in results if result["status"] == "live_evaluated")
    summary_count = len(results) - live_count
    return f"""# {title}

## Summary

- Manifest: `{checkpoint_manifest.get("manifest_path", "docs/checkpoint_manifest.json")}`
- Checkpoints compared: {len(results)}
- Live evaluated checkpoints: {live_count}
- Summary-only checkpoints: {summary_count}
- Summary-only status means the ignored local checkpoint was absent.
- Summary-only rows use committed phase summary metrics and do not claim fresh generation samples.

## Metric Definitions

{metric_lines}

## Fixed Prompt Set

{prompt_lines}

## Exact-Match Toy Tasks

{toy_lines}

## Checkpoint Reports

{checkpoint_blocks}
"""


def render_comparison_report(
    *,
    checkpoint_manifest: dict[str, Any],
    eval_config: dict[str, Any],
    results: list[dict[str, Any]],
) -> str:
    rows = "\n".join(
        "| {label} | {phase} | {status} | {params} | {loss} | {ppl} | {toksec} | {exact} | {failures} |".format(
            label=result["label"],
            phase=result["phase"],
            status=result["status"],
            params=_fmt_int(result.get("parameter_count")),
            loss=_fmt_float(result["metrics"].get("validation_loss")),
            ppl=_fmt_float(result["metrics"].get("perplexity")),
            toksec=_fmt_float(result["metrics"].get("tokens_per_sec")),
            exact=_fmt_percent(result["metrics"].get("exact_match_rate")),
            failures=", ".join(result["failure_summary"]) or "none",
        )
        for result in results
    )
    notes = "\n".join(f"- {note}" for note in checkpoint_manifest.get("comparison_notes", []))
    missing = "\n".join(
        f"- `{result['checkpoint']}`: {result['status_reason']}"
        for result in results
        if result["status"] != "live_evaluated"
    )
    if not missing:
        missing = "- None; all listed checkpoints were live evaluated."
    header = (
        "| Checkpoint | Phase | Status | Parameters | Validation loss | Perplexity | Tokens/sec | "
        "Toy exact match | Failure classes |"
    )
    return f"""# PHASE-07A Checkpoint Comparison

## Scope

- Eval config: `{checkpoint_manifest.get("eval_config", "configs/eval_fixed_prompts.yaml")}`
- Checkpoint manifest schema: {checkpoint_manifest.get("schema_version", 1)}
- Policy: {checkpoint_manifest.get("artifact_policy", "Live eval when checkpoint artifacts are present.")}

## Comparison Table

{header}
|---|---:|---:|---:|---:|---:|---:|---:|---|
{rows}

## Failure Analysis

{_render_cross_checkpoint_failures(results)}

## Missing Ignored Artifacts

{missing}

## Comparison Notes

{notes}

## Phase Gate

The evaluator uses one schema for micro, tiny, 30M+, and SFT checkpoints. When ignored checkpoints are present, it
loads and samples them directly; when they are absent, the report is explicitly marked summary-only and limited to
committed phase metrics.
"""


def _evaluate_live_checkpoint(
    *,
    entry: dict[str, Any],
    eval_config: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    kind = str(entry["kind"])
    device = resolve_device(eval_config["validation"]["device"])
    model, tokenizer, model_context = _load_model_context(entry=entry, checkpoint_path=checkpoint_path, device=device)
    metrics = _live_metrics(
        entry=entry,
        model=model,
        model_context=model_context,
        eval_config=eval_config,
        device=device,
    )
    samples = _generate_samples(
        model=model,
        tokenizer=tokenizer,
        model_context=model_context,
        eval_config=eval_config,
        entry=entry,
        device=device,
    )
    metrics.update(_generation_metrics(samples))
    metrics["perplexity"] = (
        perplexity(metrics["validation_loss"])
        if metrics.get("validation_loss") is not None
        else None
    )
    run_metrics = _read_run_metrics(Path(entry.get("run_dir", checkpoint_path.parent)))
    metrics["tokens_per_sec"] = run_metrics.get("tokens_per_sec")
    metrics["memory_usage"] = run_metrics.get("memory_usage")
    leakage = _leakage_and_memorization(entry=entry, samples=samples)
    failure_summary = _failure_summary(samples=samples, leakage=leakage)
    metadata = load_checkpoint(checkpoint_path, map_location="cpu")["metadata"]
    parameter_count = metadata.get("parameter_count") or count_parameters(model)
    return {
        "id": entry["id"],
        "label": entry["label"],
        "phase": entry.get("phase", ""),
        "kind": kind,
        "status": "live_evaluated",
        "status_reason": "Checkpoint artifact was present and loaded.",
        "checkpoint": str(checkpoint_path),
        "config": entry["config"],
        "parameter_count": parameter_count,
        "metrics": metrics,
        "samples": samples,
        "leakage": leakage,
        "failure_summary": failure_summary,
        "summary_source": "live checkpoint",
        "reproduce_command": entry.get("reproduce_command", ""),
    }


def _summary_only_result(
    *,
    entry: dict[str, Any],
    eval_config: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    summary = dict(entry["summary"])
    validation_loss = summary.get("final_validation_loss", summary.get("final_loss"))
    metrics = {
        "validation_loss": validation_loss,
        "perplexity": perplexity(float(validation_loss)) if isinstance(validation_loss, int | float) else None,
        "tokens_per_sec": summary.get("tokens_per_sec"),
        "memory_usage": summary.get("memory_usage"),
        "repetition_rate": None,
        "average_completion_tokens": None,
        "average_entropy": None,
        "exact_match_rate": summary.get("exact_match_rate"),
    }
    leakage = _summary_leakage(entry)
    failure_summary = ["live_samples_missing"]
    if summary.get("failure_summary"):
        failure_summary = list(summary["failure_summary"])
    return {
        "id": entry["id"],
        "label": entry["label"],
        "phase": entry.get("phase", ""),
        "kind": entry["kind"],
        "status": DEFAULT_SUMMARY_STATUS,
        "status_reason": "Ignored checkpoint artifact was not present in this checkout.",
        "checkpoint": str(checkpoint_path),
        "config": entry["config"],
        "parameter_count": summary.get("parameter_count"),
        "metrics": metrics,
        "samples": [],
        "leakage": leakage,
        "failure_summary": failure_summary,
        "summary_source": entry.get("summary_source", "committed checkpoint manifest summary"),
        "reproduce_command": entry.get("reproduce_command", ""),
        "prompt_count": len(eval_config["fixed_prompts"]) + len(eval_config["toy_tasks"]),
    }


def _load_model_context(
    *,
    entry: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[DecoderOnlyTransformer, Any, dict[str, Any]]:
    kind = str(entry["kind"])
    if kind == "sft":
        sft_config = load_sft_config(entry["config"])
        base_config = load_base_pretrain_config(sft_config)
        model = DecoderOnlyTransformer(base_config.model)
        load_checkpoint(checkpoint_path, model=model, map_location="cpu")
        model.to(device)
        model.eval()
        return model, load_tokenizer_for_config(base_config), {
            "kind": kind,
            "base_config": base_config,
            "sft_config": sft_config,
            "seed": sft_config.seed,
            "context_length": base_config.model.context_length,
        }
    if kind == "transformer_smoke":
        transformer_config = load_transformer_experiment_config(entry["config"])
        model = DecoderOnlyTransformer(transformer_config.model)
        load_checkpoint(checkpoint_path, model=model, map_location="cpu")
        model.to(device)
        model.eval()
        return model, load_tokenizer_for_config(transformer_config), {
            "kind": kind,
            "config": transformer_config,
            "seed": transformer_config.seed,
            "context_length": transformer_config.model.context_length,
        }
    if kind == "pretrain":
        pretrain_config = load_pretrain_config(entry["config"])
        model = DecoderOnlyTransformer(pretrain_config.model)
        load_checkpoint(checkpoint_path, model=model, map_location="cpu")
        model.to(device)
        model.eval()
        return model, load_tokenizer_for_config(pretrain_config), {
            "kind": kind,
            "config": pretrain_config,
            "seed": pretrain_config.seed,
            "context_length": pretrain_config.model.context_length,
        }
    raise ValueError(f"Unsupported checkpoint kind: {kind}")


def _live_metrics(
    *,
    entry: dict[str, Any],
    model: DecoderOnlyTransformer,
    model_context: dict[str, Any],
    eval_config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    kind = model_context["kind"]
    if kind == "sft":
        sft_config = model_context["sft_config"]
        base_config = model_context["base_config"]
        tokenizer = load_tokenizer_for_config(base_config)
        records = tuple(
            InstructionRecord(
                record_id=task["id"],
                instruction=task["instruction"],
                response=task["expected_response"],
            )
            for task in eval_config["toy_tasks"]
        )
        examples = build_sft_examples(
            records=records,
            tokenizer=tokenizer,
            template=sft_config.prompt_template,
            context_length=base_config.model.context_length,
            response_only_loss=sft_config.dataset.response_only_loss,
        )
        return {"validation_loss": evaluate_sft_loss(model=model, examples=examples, device=device)}
    if kind == "transformer_smoke":
        config = model_context["config"]
        build_tokenized_dataset_from_config(config.data.tokenized_config)
        sampler = TokenBatchSampler(
            metadata_path=config.data.metadata_path,
            split=config.data.split,
            batch_size=config.training.batch_size,
            context_length=config.model.context_length,
            seed=config.seed,
        )
        return {"validation_loss": evaluate_batch_loss(model=model, batch=sampler.next_batch(), device=device)}
    config = model_context["config"]
    build_tokenized_dataset_from_config(config.data.tokenized_config)
    sampler = TokenBatchSampler(
        metadata_path=config.data.metadata_path,
        split=config.data.validation_split,
        batch_size=config.training.batch_size,
        context_length=config.model.context_length,
        seed=config.seed + 17,
    )
    batches = [sampler.next_batch() for _ in range(eval_config["validation"]["batches"])]
    return {"validation_loss": evaluate_loss(model=model, batches=batches, device=device)}


def _generate_samples(
    *,
    model: DecoderOnlyTransformer,
    tokenizer: Any,
    model_context: dict[str, Any],
    eval_config: dict[str, Any],
    entry: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    generation = eval_config["generation"]
    prompts = list(eval_config["fixed_prompts"])
    prompts.extend(
        {
            "id": task["id"],
            "category": task.get("category", "toy_task"),
            "prompt": task["prompt"],
            "expected_response": task["expected_response"],
        }
        for task in eval_config["toy_tasks"]
    )
    rows = []
    for index, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt["prompt"], add_bos=True)
        token_ids = generate_tokens(
            model=model,
            input_ids=input_ids,
            max_new_tokens=generation["max_new_tokens"],
            seed=int(model_context["seed"]) + index,
            temperature=generation["temperature"],
            top_k=generation["top_k"],
            eos_token_id=tokenizer.eos_token_id,
            device=device,
        )
        decoded = _clean_text(tokenizer.decode(token_ids))
        completion = _completion_after_prompt(decoded=decoded, prompt=prompt["prompt"])
        distribution = _distribution_stats(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt["prompt"],
            context_length=int(model_context["context_length"]),
            top_k=generation["distribution_top_k"],
            device=device,
        )
        expected = prompt.get("expected_response")
        exact_match = _exact_match(completion, expected) if expected else None
        rows.append(
            {
                "checkpoint_id": entry["id"],
                "prompt_id": prompt["id"],
                "category": prompt["category"],
                "prompt": prompt["prompt"],
                "expected_response": expected,
                "generated_text": decoded,
                "completion": completion,
                "generated_token_count": max(len(token_ids) - len(input_ids), 0),
                "repetition_rate": repetition_rate(completion),
                "exact_match": exact_match,
                "entropy": distribution["entropy"],
                "top_tokens": distribution["top_tokens"],
            }
        )
    return rows


def _distribution_stats(
    *,
    model: DecoderOnlyTransformer,
    tokenizer: Any,
    prompt: str,
    context_length: int,
    top_k: int,
    device: torch.device,
) -> dict[str, Any]:
    token_ids = tokenizer.encode(prompt, add_bos=True)[-context_length:]
    tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(tensor).logits[0, -1].detach().cpu()
    probabilities = F.softmax(logits, dim=-1)
    entropy = float(-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum().item())
    values, indices = torch.topk(probabilities, k=min(top_k, probabilities.shape[-1]))
    return {
        "entropy": entropy,
        "top_tokens": [
            {
                "token_id": int(token_id),
                "probability": float(probability),
                "decoded": _clean_text(tokenizer.decode([int(token_id)])),
            }
            for probability, token_id in zip(values.tolist(), indices.tolist(), strict=True)
        ],
    }


def repetition_rate(text: str, *, ngram_size: int = 3) -> float:
    normalized = normalize_text(text)
    if not normalized:
        return 0.0
    tokens = normalized.split()
    if len(tokens) < ngram_size + 1:
        tokens = list(normalized)
    if len(tokens) < ngram_size + 1:
        return 0.0
    ngrams = [tuple(tokens[index : index + ngram_size]) for index in range(len(tokens) - ngram_size + 1)]
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def _generation_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {
            "repetition_rate": None,
            "average_completion_tokens": None,
            "average_entropy": None,
            "exact_match_rate": None,
        }
    exact_values = [sample["exact_match"] for sample in samples if sample["exact_match"] is not None]
    return {
        "repetition_rate": sum(float(sample["repetition_rate"]) for sample in samples) / len(samples),
        "average_completion_tokens": sum(int(sample["generated_token_count"]) for sample in samples) / len(samples),
        "average_entropy": sum(float(sample["entropy"]) for sample in samples) / len(samples),
        "exact_match_rate": sum(1 for value in exact_values if value) / len(exact_values) if exact_values else None,
    }


def _leakage_and_memorization(*, entry: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    references = _reference_texts(entry)
    generated = "\n".join(sample["completion"] for sample in samples)
    normalized_generated = normalize_text(generated)
    memorized = [
        reference
        for reference in references
        if len(reference) >= 16 and reference in normalized_generated
    ]
    manifest_leakage = _summary_leakage(entry)
    return {
        "reference_count": len(references),
        "memorized_fragment_count": len(memorized),
        "memorized_fragments": memorized[:5],
        "manifest_leakage": manifest_leakage,
    }


def _summary_leakage(entry: dict[str, Any]) -> dict[str, Any]:
    leakage_rows = []
    for manifest_path in entry.get("data_manifests", []):
        path = Path(manifest_path)
        if not path.is_file():
            leakage_rows.append({"manifest": str(path), "status": "missing"})
            continue
        payload = json.loads(path.read_text(encoding="utf8"))
        leakage_check = payload.get("leakage_check")
        if not isinstance(leakage_check, dict):
            leakage_rows.append(
                {
                    "manifest": str(path),
                    "overlap_count": "not_reported",
                    "method": "manifest does not expose a train/validation leakage_check field",
                }
            )
            continue
        leakage_rows.append(
            {
                "manifest": str(path),
                "overlap_count": leakage_check.get("overlap_count"),
                "method": leakage_check.get("method"),
            }
        )
    return {"data_manifests": leakage_rows}


def _reference_texts(entry: dict[str, Any]) -> list[str]:
    kind = str(entry["kind"])
    if kind == "sft":
        config = load_sft_config(entry["config"])
        return [normalize_text(f"{record.instruction}\n{record.response}") for record in config.dataset.records]
    try:
        if kind == "transformer_smoke":
            config = load_transformer_experiment_config(entry["config"])
            raw = load_yaml_config(config.data.tokenized_config)
        else:
            config = load_pretrain_config(entry["config"])
            raw = load_yaml_config(config.data.tokenized_config)
        return [normalize_text(record.text) for record in token_records_from_config(raw)]
    except (FileNotFoundError, ValueError, KeyError):
        return []


def _failure_summary(*, samples: list[dict[str, Any]], leakage: dict[str, Any]) -> list[str]:
    failures: set[str] = set()
    completions = [normalize_text(sample["completion"]) for sample in samples]
    if len(set(completions)) <= max(1, len(completions) // 2):
        failures.add("mode_collapse")
    if leakage["memorized_fragment_count"] > 0:
        failures.add("memorized_fragments")
    for sample in samples:
        completion = normalize_text(sample["completion"])
        if not completion:
            failures.add("pure_gibberish")
        if sample["repetition_rate"] >= 0.25:
            failures.add("repetition_loop")
        if "�" in completion:
            failures.add("bad_token_boundaries")
        if _has_latin(completion) and _has_japanese(completion) and sample["category"] != "bilingual":
            failures.add("language_mixing")
        if sample["exact_match"] is False:
            failures.add("instruction_ignored")
        if re.search(r"\b(certainly|definitely|guaranteed|always)\b", completion, flags=re.IGNORECASE):
            failures.add("false_factual_confidence")
        if completion and len(completion.split()) >= 4 and not re.search(r"[.!?。！？]$", completion):
            failures.add("syntax_without_semantics")
    return sorted(failures)


def _render_checkpoint_result(result: dict[str, Any]) -> str:
    sample_block = _render_samples(result["samples"])
    leakage = result["leakage"]
    manifest_leakage = leakage.get("manifest_leakage", leakage).get("data_manifests", [])
    leakage_lines = "\n".join(
        "- `{manifest}`: overlap_count={overlap} ({method})".format(
            manifest=row["manifest"],
            overlap=row.get("overlap_count", "n/a"),
            method=row.get("method", row.get("status")),
        )
        for row in manifest_leakage
    )
    if not leakage_lines:
        leakage_lines = "- No data manifest evidence was available for this entry."
    return f"""### {result["label"]}

- Phase: {result["phase"]}
- Kind: {result["kind"]}
- Status: {result["status"]}
- Status reason: {result["status_reason"]}
- Checkpoint: `{result["checkpoint"]}`
- Parameters: {_fmt_int(result.get("parameter_count"))}
- Validation loss: {_fmt_float(result["metrics"].get("validation_loss"))}
- Perplexity: {_fmt_float(result["metrics"].get("perplexity"))}
- Tokens/sec: {_fmt_float(result["metrics"].get("tokens_per_sec"))}
- Repetition rate: {_fmt_float(result["metrics"].get("repetition_rate"))}
- Average entropy: {_fmt_float(result["metrics"].get("average_entropy"))}
- Toy exact match: {_fmt_percent(result["metrics"].get("exact_match_rate"))}
- Failure classes: {", ".join(result["failure_summary"]) or "none"}
- Reproduce command: `{result.get("reproduce_command", "")}`

Leakage and memorization:
{leakage_lines}
- Memorized generated fragments: {leakage.get("memorized_fragment_count", "n/a")}

{sample_block}
"""


def _render_samples(samples: list[dict[str, Any]]) -> str:
    if not samples:
        return "Samples: not generated because the checkpoint artifact was absent."
    rows = []
    for sample in samples:
        top_tokens = ", ".join(
            f"{item['decoded'] or '<blank>'}:{item['probability']:.3f}" for item in sample["top_tokens"][:3]
        )
        rows.append(
            f"- `{sample['prompt_id']}` {sample['category']}: completion=`{sample['completion']}`; "
            f"repetition={sample['repetition_rate']:.3f}; entropy={sample['entropy']:.3f}; top={top_tokens}"
        )
    return "Samples:\n" + "\n".join(rows)


def _render_cross_checkpoint_failures(results: list[dict[str, Any]]) -> str:
    lines = []
    for result in results:
        failures = ", ".join(result["failure_summary"]) or "none"
        lines.append(f"- {result['label']}: {failures}")
    return "\n".join(lines)


def _render_metric_definitions(eval_config: dict[str, Any]) -> str:
    definitions = eval_config.get("metric_definitions") or {}
    if not definitions:
        definitions = {
            "validation_loss": "Mean next-token or response-only cross entropy on the configured validation probe.",
            "perplexity": "exp(validation_loss), capped internally the same way as training reports.",
            "tokens_per_sec": "Last recorded training/eval run throughput when a run metrics file is present.",
            "repetition_rate": "Repeated 3-gram fraction over generated completion text.",
            "exact_match_rate": "Fraction of toy instruction completions matching the expected response exactly.",
        }
    return "\n".join(f"- `{key}`: {value}" for key, value in definitions.items())


def _read_run_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        return {"tokens_per_sec": None, "memory_usage": None}
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf8").splitlines()
        if line.strip()
    ]
    if not rows:
        return {"tokens_per_sec": None, "memory_usage": None}
    last = rows[-1]
    return {
        "tokens_per_sec": last.get("tokens_per_sec"),
        "memory_usage": last.get("memory_usage"),
    }


def _normalize_fixed_prompts(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("fixed_prompts must be a non-empty list.")
    prompts = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            prompts.append({"id": f"prompt-{index + 1:03d}", "category": "legacy", "prompt": item})
        elif isinstance(item, dict):
            prompts.append(
                {
                    "id": str(item.get("id", f"prompt-{index + 1:03d}")),
                    "category": str(item.get("category", "general")),
                    "prompt": _required_str(item, "prompt"),
                }
            )
        else:
            raise ValueError("fixed_prompts entries must be strings or mappings.")
    return prompts


def _normalize_toy_tasks(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("toy_tasks must be a list.")
    tasks = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError("toy_tasks entries must be mappings.")
        instruction = _required_str(item, "instruction")
        prompt = str(item.get("prompt", f"Q:{instruction}\nA:"))
        tasks.append(
            {
                "id": str(item.get("id", f"toy-{index + 1:03d}")),
                "category": str(item.get("category", "toy_task")),
                "instruction": instruction,
                "prompt": prompt,
                "expected_response": _required_str(item, "expected_response"),
            }
        )
    return tasks


def _completion_after_prompt(*, decoded: str, prompt: str) -> str:
    if decoded.startswith(prompt):
        return decoded[len(prompt) :].strip()
    normalized_decoded = normalize_text(decoded)
    normalized_prompt = normalize_text(prompt)
    if normalized_decoded.startswith(normalized_prompt):
        return normalized_decoded[len(normalized_prompt) :].strip()
    return normalized_decoded.strip()


def _exact_match(completion: str, expected: str | None) -> bool:
    if expected is None:
        return False
    return normalize_text(completion).split("<eos>", maxsplit=1)[0].strip() == normalize_text(expected)


def _clean_text(text: str) -> str:
    return text.replace("\x00", "").strip()


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing string field: {key}")
    return value


def _has_japanese(text: str) -> bool:
    return any("\u3040" <= char <= "\u30ff" or "\u4e00" <= char <= "\u9fff" for char in text)


def _has_latin(text: str) -> bool:
    return any("a" <= char.lower() <= "z" for char in text)


def _fmt_float(value: Any) -> str:
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return f"{float(value):.4f}"
    return "n/a"


def _fmt_percent(value: Any) -> str:
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return f"{float(value):.2%}"
    return "n/a"


def _fmt_int(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{int(value):,}"
    return "n/a"
