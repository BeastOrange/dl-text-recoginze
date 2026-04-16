from pathlib import Path

from dltr.models.recognition.charset import CharacterVocabulary


def test_character_vocabulary_encodes_and_decodes_text(tmp_path: Path) -> None:
    charset_path = tmp_path / "charset.txt"
    charset_path.write_text("营\n业\n时\n间\nA\n1\n:\n", encoding="utf-8")

    vocab = CharacterVocabulary.from_file(charset_path)
    encoded = vocab.encode("营业A1")
    decoded = vocab.decode(encoded)

    assert vocab.blank_index == 0
    assert encoded[0] != 0
    assert decoded == "营业A1"


def test_character_vocabulary_skips_unknown_characters(tmp_path: Path) -> None:
    charset_path = tmp_path / "charset.txt"
    charset_path.write_text("营\n业\n", encoding="utf-8")

    vocab = CharacterVocabulary.from_file(charset_path)

    assert vocab.encode("营业X") == [1, 2]


def test_character_vocabulary_reports_oov_count(tmp_path: Path) -> None:
    charset_path = tmp_path / "charset.txt"
    charset_path.write_text("A\nB\n1\n", encoding="utf-8")

    vocab = CharacterVocabulary.from_file(charset_path)
    encoded, oov_count = vocab.encode_with_oov_count("AB?1x")

    assert encoded == [1, 2, 3]
    assert oov_count == 2
