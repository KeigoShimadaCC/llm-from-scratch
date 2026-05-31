import pytest

from kgpt.char_tokenizer import CharacterTokenizer


def test_character_tokenizer_roundtrip_and_serialization(tmp_path) -> None:
    text = "hello microgpt.\n"
    tokenizer = CharacterTokenizer.build([text], tokenizer_id="pytest-char")

    token_ids = tokenizer.encode(text)
    assert tokenizer.decode(token_ids) == text
    assert tokenizer.vocab_size == len(set(text))

    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    restored = CharacterTokenizer.load(tokenizer_path)

    assert restored.to_dict() == tokenizer.to_dict()
    assert restored.decode(restored.encode(text)) == text


def test_character_tokenizer_rejects_unknown_characters() -> None:
    tokenizer = CharacterTokenizer.build(["abc"], tokenizer_id="pytest-char")

    with pytest.raises(ValueError, match="outside the vocabulary"):
        tokenizer.encode("abcd")
