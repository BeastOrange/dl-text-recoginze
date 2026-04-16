from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CharacterVocabulary:
    characters: tuple[str, ...]

    @property
    def blank_index(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return len(self.characters) + 1

    @classmethod
    def from_file(cls, charset_path: str | Path) -> CharacterVocabulary:
        path = Path(charset_path)
        characters = tuple(
            line.strip("\n")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip("\n")
        )
        if not characters:
            raise ValueError(f"Charset file is empty: {path}")
        return cls(characters=characters)

    def encode(self, text: str) -> list[int]:
        index_map = {character: index + 1 for index, character in enumerate(self.characters)}
        return [index_map[character] for character in text if character in index_map]

    def encode_with_oov_count(self, text: str) -> tuple[list[int], int]:
        index_map = {character: index + 1 for index, character in enumerate(self.characters)}
        encoded: list[int] = []
        oov_count = 0
        for character in text:
            index = index_map.get(character)
            if index is None:
                oov_count += 1
                continue
            encoded.append(index)
        return encoded, oov_count

    def decode(self, indices: list[int]) -> str:
        decoded: list[str] = []
        for index in indices:
            if index <= 0:
                continue
            if index > len(self.characters):
                continue
            decoded.append(self.characters[index - 1])
        return "".join(decoded)

    def decode_greedy(self, indices: list[int]) -> str:
        deduplicated: list[int] = []
        previous = None
        for index in indices:
            if index == previous:
                continue
            previous = index
            if index == self.blank_index:
                continue
            deduplicated.append(index)
        return self.decode(deduplicated)
