from __future__ import annotations

import hashlib
import re
from pathlib import Path

import deal


def clean_text(text: str) -> str:
    """Normalize whitespace while keeping line-oriented grammar/code blocks readable."""
    normalized = text.replace("\xa0", " ")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n[ \t]+", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    return normalized.strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def short_sha256(text: str, *, length: int = 16) -> str:
    """Return first `length` characters of SHA-256 hex digest."""
    if not 1 <= length <= 64:
        raise ValueError("length must be between 1 and 64")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


@deal.pre(lambda section_index, chunk_index, anchor, text: section_index >= 0)
@deal.pre(lambda section_index, chunk_index, anchor, text: chunk_index >= 0)
def stable_chunk_id(section_index: int, chunk_index: int, anchor: str, text: str) -> str:
    digest = short_sha256(f"{anchor}:{chunk_index}:{text}")
    return f"go-spec-{section_index:04d}-{chunk_index:03d}-{digest}"


@deal.pre(lambda section_index, anchor, title: section_index >= 0)
def stable_section_id(section_index: int, anchor: str, title: str) -> str:
    digest = short_sha256(f"{anchor}:{title}", length=8)
    return f"go-spec-section-{section_index:04d}-{digest}"


def relative_display_path(path: Path, *, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


@deal.pre(lambda value, minimum, maximum: minimum <= maximum)
@deal.ensure(lambda value, minimum, maximum, result: minimum <= result <= maximum)
def bounded_int(value: int, minimum: int, maximum: int) -> int:
    return min(max(value, minimum), maximum)
