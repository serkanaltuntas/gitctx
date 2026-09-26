"""Lossless byte-level BPE with structural IDs outside content tokenization."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL = {name: i for i, name in enumerate(
    ("<pad>", "<bos>", "<eos>", "<system>", "<user>", "<assistant>", "<sep>"))}
OFFSET = len(SPECIAL)
VERSION = "byte-bpe-student-v1"
CHUNK_PATTERN = r"[\s\S]{1,1024}"


def content_pre_tokenizer():
    # Bound pathological minified/generated lines without dropping any character.
    return pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(CHUNK_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])


class StudentTokenizer:
    """Literal marker strings remain content; roles are appended as integer IDs."""

    def __init__(self, backend: Tokenizer):
        self.backend = backend
        self._vocab_size = backend.get_vocab_size() + OFFSET
        config = json.loads(backend.to_str())
        if (config["normalizer"] is not None or config["added_tokens"]
                or config["model"]["type"] != "BPE"
                or config["model"]["unk_token"] is not None
                or config["model"]["dropout"] is not None
                or config["post_processor"] is not None
                or config["truncation"] is not None or config["padding"] is not None
                or config["pre_tokenizer"] != json.loads(content_pre_tokenizer().__getstate__())
                or config["decoder"] != {
                    "type": "ByteLevel", "add_prefix_space": True,
                    "trim_offsets": True, "use_regex": True}):
            raise ValueError("tokenizer violates the lossless content contract")
        if not set(pre_tokenizers.ByteLevel.alphabet()) <= set(backend.get_vocab()):
            raise ValueError("tokenizer must contain the complete byte alphabet")
        # Invert the standard reversible ByteLevel byte-to-character alphabet.
        visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        remaining = [b for b in range(256) if b not in visible]
        byte_map = {chr(b): b for b in visible}
        byte_map.update({chr(256 + i): b for i, b in enumerate(remaining)})
        if set(byte_map) != set(pre_tokenizers.ByteLevel.alphabet()):
            raise ValueError("unexpected upstream byte alphabet")
        self._bytes = [bytes(byte_map[c] for c in backend.id_to_token(i))
                       for i in range(backend.get_vocab_size())]

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @classmethod
    def fit(cls, texts: Iterable[str], vocab_size: int = 32000) -> StudentTokenizer:
        if vocab_size < 256 + OFFSET:
            raise ValueError("vocabulary cannot fit the byte alphabet and structural IDs")
        backend = Tokenizer(models.BPE())
        backend.pre_tokenizer = content_pre_tokenizer()
        backend.decoder = decoders.ByteLevel()
        backend.train_from_iterator(texts, trainer=trainers.BpeTrainer(
            vocab_size=vocab_size - OFFSET, min_frequency=2, show_progress=False,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=[]))
        return cls(backend)

    def encode(self, text: str) -> list[int]:
        text.encode("utf-8", errors="strict")
        return [i + OFFSET for i in self.backend.encode(text, add_special_tokens=False).ids]

    def decode(self, ids: list[int]) -> str:
        if any(i < OFFSET or i >= self.vocab_size for i in ids):
            raise ValueError("content decoder received a structural or invalid token ID")
        # Join byte pieces once and use a cached vocabulary bound. Decode UTF-8 only
        # after joining, since an individual token may end inside a code point.
        return b"".join(self._bytes[i - OFFSET] for i in ids).decode("utf-8", errors="strict")

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "version": VERSION, "special_tokens": SPECIAL,
            "content_id_offset": OFFSET, "vocab_size": self.vocab_size,
            "backend": json.loads(self.backend.to_str()),
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> StudentTokenizer:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (value["version"] != VERSION or value["special_tokens"] != SPECIAL
                or value["content_id_offset"] != OFFSET):
            raise ValueError("unsupported tokenizer identity")
        result = cls(Tokenizer.from_str(json.dumps(value["backend"])))
        if result.vocab_size != value["vocab_size"]:
            raise ValueError("vocabulary size mismatch")
        return result
