from __future__ import annotations


def main() -> int:
    print(
        "Use `python -m tokenizer.train_report "
        "--config configs/tokenizer_bilingual.yaml --output docs/tokenizer_report.md`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
