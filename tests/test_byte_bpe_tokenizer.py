from kgpt.byte_bpe import ByteBPETokenizer, train_byte_bpe


def test_byte_bpe_roundtrip_save_load_and_byte_fallback(tmp_path) -> None:
    text = "hello tokenizer\n小さなモデル 😀\n"
    tokenizer = train_byte_bpe(
        [text, text],
        tokenizer_id="pytest-byte-bpe",
        target_vocab_size=300,
        min_pair_frequency=2,
    )

    token_ids = tokenizer.encode(text)
    assert tokenizer.decode(token_ids) == text
    assert tokenizer.count_unknowns(token_ids) == 0
    assert tokenizer.decode(tokenizer.encode("unseen 𠮷🧪")) == "unseen 𠮷🧪"

    tokenizer_path = tmp_path / "byte_bpe.json"
    tokenizer.save(tokenizer_path)
    restored = ByteBPETokenizer.load(tokenizer_path)

    assert restored.to_dict() == tokenizer.to_dict()
    assert restored.decode(restored.encode(text)) == text
