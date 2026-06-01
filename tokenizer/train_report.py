from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kgpt.byte_bpe import ByteBPETokenizer, display_piece, train_byte_bpe
from kgpt.config import file_sha256
from kgpt.token_data import load_yaml_config, records_from_config


@dataclass(frozen=True)
class SentenceStats:
    text: str
    token_count: int
    utf8_bytes: int
    character_count: int
    unknown_count: int
    compression_ratio: float


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train byte-level BPE candidates and write PHASE-02A tokenizer report."
    )
    parser.add_argument("--config", required=True, help="Path to tokenizer report YAML config.")
    parser.add_argument("--output", required=True, help="Path to write markdown report.")
    args = parser.parse_args(argv)

    report = generate_tokenizer_report(config_path=Path(args.config), output_path=Path(args.output))
    print(
        json.dumps(
            {
                "output": str(report["output_path"]),
                "tokenizer_model": str(report["tokenizer_model_path"]),
                "selected_vocab_size": report["selected_tokenizer"].vocab_size,
            },
            sort_keys=True,
        )
    )
    return 0


def generate_tokenizer_report(*, config_path: Path, output_path: Path) -> dict[str, Any]:
    raw = load_yaml_config(config_path)
    tokenizer_config = _required_mapping(raw, "tokenizer")
    report_config = _required_mapping(raw, "report")
    records = records_from_config(raw)
    record_texts = [record.text for record in records]
    english_texts = [record.text for record in records if record.language.startswith("en")]
    if not english_texts:
        raise ValueError("At least one English record is required for tokenizer comparison.")

    target_vocab_sizes = _required_int_list(tokenizer_config, "target_vocab_sizes")
    selected_target_vocab_size = _required_int(tokenizer_config, "selected_target_vocab_size")
    min_pair_frequency = _required_int(tokenizer_config, "min_pair_frequency")
    tokenizer_id = _required_str(tokenizer_config, "tokenizer_id")

    sweeps: dict[str, list[dict[str, Any]]] = {
        "english_only": [],
        "bilingual": [],
    }
    trained_tokenizers: dict[tuple[str, int], ByteBPETokenizer] = {}
    for corpus_name, texts in (("english_only", english_texts), ("bilingual", record_texts)):
        for target_vocab_size in target_vocab_sizes:
            candidate = train_byte_bpe(
                texts,
                tokenizer_id=f"{tokenizer_id}-{corpus_name}-{target_vocab_size}",
                target_vocab_size=target_vocab_size,
                min_pair_frequency=min_pair_frequency,
            )
            trained_tokenizers[(corpus_name, target_vocab_size)] = candidate
            sweeps[corpus_name].append(
                {
                    "requested_vocab_size": target_vocab_size,
                    "actual_vocab_size": candidate.vocab_size,
                    "merge_count": candidate.merge_count,
                    "status": _sweep_status(candidate, target_vocab_size, min_pair_frequency),
                }
            )

    selected = trained_tokenizers[("bilingual", selected_target_vocab_size)]
    model_path = Path(_required_str(tokenizer_config, "model_path"))
    selected = ByteBPETokenizer(tokenizer_id=tokenizer_id, merges=selected.merges)
    selected.save(model_path)

    sentence_groups = _sentence_groups(report_config)
    selected_group_stats = {
        group_name: [_sentence_stats(selected, sentence) for sentence in sentences]
        for group_name, sentences in sentence_groups.items()
    }
    english_only = trained_tokenizers[("english_only", selected_target_vocab_size)]
    comparison_stats = {
        "english_only": {
            group_name: [_sentence_stats(english_only, sentence) for sentence in sentences]
            for group_name, sentences in sentence_groups.items()
        },
        "bilingual": selected_group_stats,
    }
    fallback_probes = _required_str_list(report_config, "byte_fallback_probes")
    bad_examples = _required_str_list(report_config, "bad_segmentation_examples")
    context_lengths = _required_int_list(report_config, "context_lengths")
    candidate_notes = _candidate_notes(tokenizer_config)

    markdown = _render_markdown(
        config_path=config_path,
        output_path=output_path,
        report_title=str(report_config.get("title", "PHASE-02A Tokenizer Report")),
        phase_note=str(
            report_config.get(
                "phase_note",
                "The final tokenizer choice for larger training is deferred. This phase only proves the "
                "token-level pipeline on a repo-authored bilingual smoke corpus; larger approved data is "
                "required before PHASE-04A/PHASE-05A should treat this tokenizer as final.",
            )
        ),
        model_path=model_path,
        selected=selected,
        sweeps=sweeps,
        sentence_groups=sentence_groups,
        selected_group_stats=selected_group_stats,
        comparison_stats=comparison_stats,
        fallback_probes=fallback_probes,
        bad_examples=bad_examples,
        context_lengths=context_lengths,
        candidate_notes=candidate_notes,
        min_pair_frequency=min_pair_frequency,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf8")
    return {
        "output_path": output_path,
        "tokenizer_model_path": model_path,
        "selected_tokenizer": selected,
        "sweeps": sweeps,
    }


def _render_markdown(
    *,
    config_path: Path,
    output_path: Path,
    report_title: str,
    phase_note: str,
    model_path: Path,
    selected: ByteBPETokenizer,
    sweeps: dict[str, list[dict[str, Any]]],
    sentence_groups: dict[str, list[str]],
    selected_group_stats: dict[str, list[SentenceStats]],
    comparison_stats: dict[str, dict[str, list[SentenceStats]]],
    fallback_probes: list[str],
    bad_examples: list[str],
    context_lengths: list[int],
    candidate_notes: list[dict[str, str]],
    min_pair_frequency: int,
) -> str:
    lines: list[str] = [
        f"# {report_title}",
        "",
        "## Summary",
        "",
        f"- Selected tokenizer: `{selected.tokenizer_id}`.",
        "- Algorithm: repo-owned byte-level BPE with full UTF-8 byte fallback.",
        f"- Vocabulary size: {selected.vocab_size} tokens "
        f"({selected.merge_count} learned merges plus 256 byte tokens and special tokens).",
        f"- Config: `{config_path}`.",
        f"- Report output: `{output_path}`.",
        f"- Ignored tokenizer model artifact: `{model_path}`.",
        f"- Tokenizer model sha256: `{file_sha256(model_path)}`.",
        "",
        phase_note,
        "",
        "## Candidate Status",
        "",
        "| Candidate | Status | Rationale |",
        "| --- | --- | --- |",
    ]
    for note in candidate_notes:
        lines.append(f"| `{note['id']}` | {note['status']} | {note['rationale']} |")

    lines.extend(
        [
            "",
            "## Vocabulary Sweep",
            "",
            f"Minimum pair frequency: {min_pair_frequency}. Requested vocabulary sizes come from the phase config,",
            "but smoke corpora are intentionally small, so byte-level BPE stops when no more eligible pairs remain.",
            "",
            "| Corpus | Requested vocab | Actual vocab | Learned merges | Status |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for corpus_name, rows in sweeps.items():
        for row in rows:
            lines.append(
                "| "
                f"{corpus_name} | {row['requested_vocab_size']} | {row['actual_vocab_size']} | "
                f"{row['merge_count']} | {row['status']} |"
            )

    lines.extend(
        [
            "",
            "## English/Japanese Tokenization",
            "",
            "| Group | Sentences | Mean tokens/sentence | Mean bytes/token | Mean unknown tokens |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for group_name, stats in selected_group_stats.items():
        lines.append(
            "| "
            f"{group_name} | {len(stats)} | {_mean([item.token_count for item in stats]):.2f} | "
            f"{_mean([item.compression_ratio for item in stats]):.2f} | "
            f"{_mean([item.unknown_count for item in stats]):.2f} |"
        )

    lines.extend(
        [
            "",
            "## English-Only vs Bilingual Comparison",
            "",
            "| Tokenizer corpus | Sentence group | Mean tokens/sentence | Mean bytes/token |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for corpus_name, groups in comparison_stats.items():
        for group_name, stats in groups.items():
            lines.append(
                "| "
                f"{corpus_name} | {group_name} | {_mean([item.token_count for item in stats]):.2f} | "
                f"{_mean([item.compression_ratio for item in stats]):.2f} |"
            )

    lines.extend(
        [
            "",
            "## Unknown And Byte Fallback Behavior",
            "",
            "The tokenizer reserves `<unk>`, but normal text encoding uses UTF-8 bytes, so unseen Unicode "
            "still roundtrips without producing unknown tokens.",
            "",
            "| Probe | Tokens | Unknowns | Roundtrip |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for probe in fallback_probes:
        token_ids = selected.encode(probe)
        lines.append(
            f"| {_escape_table(probe)} | {len(token_ids)} | {selected.count_unknowns(token_ids)} | "
            f"{'pass' if selected.decode(token_ids) == probe else 'fail'} |"
        )

    lines.extend(
        [
            "",
            "## Compression And Context-Length Effect",
            "",
            "Bytes/token is a practical compression proxy: higher values mean each context window carries "
            "more source text.",
            "",
            "| Sentence group | Context tokens | Approx UTF-8 bytes/context | Approx characters/context |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for group_name, stats in selected_group_stats.items():
        mean_bytes_per_token = _mean([item.compression_ratio for item in stats])
        mean_chars_per_token = _mean(
            [item.character_count / item.token_count for item in stats if item.token_count > 0]
        )
        for context_length in context_lengths:
            lines.append(
                "| "
                f"{group_name} | {context_length} | {context_length * mean_bytes_per_token:.1f} | "
                f"{context_length * mean_chars_per_token:.1f} |"
            )

    lines.extend(
        [
            "",
            "## Failure Cases And Segmentation Examples",
            "",
            "These examples are expected rough edges for a tiny smoke corpus. Japanese text and emoji-heavy "
            "mixed text still fall back to byte fragments when the corpus has not repeated the same byte "
            "patterns enough to merge them.",
            "",
        ]
    )
    for example in bad_examples:
        pieces = selected.token_pieces(example)
        preview = ", ".join(display_piece(piece) for piece in pieces[:32])
        if len(pieces) > 32:
            preview += ", ..."
        stats = _sentence_stats(selected, example)
        lines.extend(
            [
                f"- `{_inline_code(example)}`",
                f"  - tokens: {stats.token_count}; bytes/token: {stats.compression_ratio:.2f}; pieces: {preview}",
            ]
        )

    lines.extend(
        [
            "",
            "## Sentence Samples",
            "",
            "| Group | Sentence | Tokens | Bytes/token |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for group_name, sentences in sentence_groups.items():
        for sentence in sentences:
            stats = _sentence_stats(selected, sentence)
            lines.append(
                f"| {group_name} | {_escape_table(sentence)} | {stats.token_count} | {stats.compression_ratio:.2f} |"
            )

    lines.append("")
    return "\n".join(lines)


def _sentence_stats(tokenizer: ByteBPETokenizer, sentence: str) -> SentenceStats:
    token_ids = tokenizer.encode(sentence)
    token_count = len(token_ids)
    utf8_bytes = len(sentence.encode("utf8"))
    return SentenceStats(
        text=sentence,
        token_count=token_count,
        utf8_bytes=utf8_bytes,
        character_count=len(sentence),
        unknown_count=tokenizer.count_unknowns(token_ids),
        compression_ratio=utf8_bytes / max(token_count, 1),
    )


def _sentence_groups(report_config: dict[str, Any]) -> dict[str, list[str]]:
    sentences = _required_mapping(report_config, "sentences")
    return {
        "english": _required_str_list(sentences, "english"),
        "japanese": _required_str_list(sentences, "japanese"),
        "mixed": _required_str_list(sentences, "mixed"),
    }


def _candidate_notes(tokenizer_config: dict[str, Any]) -> list[dict[str, str]]:
    candidates = tokenizer_config.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("tokenizer.candidates must be a non-empty list.")
    notes: list[dict[str, str]] = []
    for item in candidates:
        if not isinstance(item, dict):
            raise ValueError("tokenizer.candidates entries must be mappings.")
        notes.append(
            {
                "id": _required_str(item, "id"),
                "status": _required_str(item, "status"),
                "rationale": _required_str(item, "rationale"),
            }
        )
    return notes


def _sweep_status(tokenizer: ByteBPETokenizer, requested_vocab_size: int, min_pair_frequency: int) -> str:
    if tokenizer.vocab_size >= requested_vocab_size:
        return "trained to requested size"
    return f"capped; no more pairs met min frequency {min_pair_frequency}"


def _mean(values: list[int] | list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _inline_code(value: str) -> str:
    return value.replace("`", "\\`")


def _required_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing mapping config field: {key}")
    return value


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing string config field: {key}")
    return value


def _required_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Missing integer config field: {key}")
    return value


def _required_int_list(raw: dict[str, Any], key: str) -> list[int]:
    value = raw.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, int) for item in value):
        raise ValueError(f"Missing integer list config field: {key}")
    return value


def _required_str_list(raw: dict[str, Any], key: str) -> list[str]:
    value = raw.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"Missing string list config field: {key}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
