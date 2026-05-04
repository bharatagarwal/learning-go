from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import deal
from rank_bm25 import BM25L

from scripts.go_spec_rag.corpus import Corpus
from scripts.go_spec_rag.models import ChunkRecord

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")

SPEC_TERM_EXPANSIONS: dict[str, str] = {
    "array": "array type array length element type distinct storage comparable",
    "arrays": "array type array length element type distinct storage comparable",
    "assignable": "assignability assignable assignment type identical underlying type",
    "assignment": "assignability assignable assignment type identical underlying type",
    "cap": "capacity len cap slice array channel",
    "capacity": "capacity len cap slice array channel underlying array",
    "comparable": "comparison operators comparable strictly comparable equality nil",
    "compare": "comparison operators comparable strictly comparable equality nil",
    "constraint": "type constraint type set interface comparable satisfies implements",
    "constraints": "type constraint type set interface comparable satisfies implements",
    "conversion": "conversion convert assignability underlying type representable",
    "conversions": "conversion convert assignability underlying type representable",
    "generic": "type parameter type argument constraint type set inference instantiation",
    "generics": "type parameter type argument constraint type set inference instantiation",
    "interface": "interface type type set method set implements comparable basic interface",
    "interfaces": "interface type type set method set implements comparable basic interface",
    "len": "length len cap slice array map channel string",
    "length": "length len cap slice array map channel string",
    "map": "map type key element comparable nil length make",
    "maps": "map type key element comparable nil length make",
    "method": "method set receiver interface implements",
    "methods": "method set receiver interface implements",
    "nil": "nil predeclared identifier zero value pointer slice map channel function interface",
    "slice": "slice type underlying array length capacity nil make",
    "slices": "slice type underlying array length capacity nil make",
    "string": "string type string literal rune byte conversion",
    "strings": "string type string literal rune byte conversion",
    "struct": "struct type field comparable blank field",
    "structs": "struct type field comparable blank field",
    "type": "type identity underlying type defined type alias type parameter",
    "types": "type identity underlying type defined type alias type parameter",
}


@dataclass(frozen=True)
class LexicalHit:
    chunk_id: str
    score: float


@deal.ensure(lambda text, result: all(isinstance(t, str) and len(t) > 0 for t in result))
def tokenize(text: str) -> list[str]:
    """Split text into lowercased, normalized tokens."""
    return [normalize_token(match.group(0)) for match in TOKEN_RE.finditer(text)]


@deal.pre(lambda token: len(token) > 0)
@deal.ensure(lambda token, result: len(result) > 0)
def normalize_token(token: str) -> str:
    """Lowercase and strip common English plural suffixes."""
    lowered = token.lower()
    if len(lowered) > 4 and lowered.endswith("ies"):
        return lowered[:-3] + "y"
    if len(lowered) > 3 and lowered.endswith("s"):
        return lowered[:-1]
    return lowered


@deal.ensure(lambda query, result: len(result) >= 1 and len(result) <= 8)
def query_variants(query: str) -> list[str]:
    stripped = query.strip()
    variants: list[str] = [stripped] if stripped else [query]
    tokens = tokenize(query)
    if tokens:
        _append_keyword_variant(variants, tokens, stripped)
        _append_expansion_variants(variants, tokens, stripped or query)
    return _deduplicate_variants(variants)[:8]


def _append_keyword_variant(variants: list[str], tokens: list[str], stripped: str) -> None:
    keyword_query = " ".join(tokens)
    if keyword_query and keyword_query != stripped:
        variants.append(keyword_query)


def _append_expansion_variants(variants: list[str], tokens: list[str], base_query: str) -> None:
    expansions = [SPEC_TERM_EXPANSIONS[token] for token in tokens if token in SPEC_TERM_EXPANSIONS]
    for expansion in expansions:
        variants.append(expansion)
    if expansions:
        variants.append(" ".join([base_query, *expansions]))


def _deduplicate_variants(variants: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        normalized = " ".join(variant.split())
        if normalized:
            if normalized not in seen:
                unique.append(normalized)
                seen.add(normalized)
        elif not unique:
            unique.append(variant)
            seen.add(variant)
    return unique


def lexical_search(corpus: Corpus, variants: Sequence[str], *, limit: int) -> list[LexicalHit]:
    if limit < 1:
        raise ValueError("limit must be positive")
    if not corpus.chunks:
        return []

    tokenized_docs = [chunk_tokens(chunk) for chunk in corpus.chunks]
    bm25 = BM25L(tokenized_docs)

    scores = [0.0 for _ in corpus.chunks]
    for variant in variants:
        variant_scores = bm25.get_scores(tokenize(variant))
        for index, score in enumerate(variant_scores):
            scores[index] = max(scores[index], float(score))

    max_score = max(scores, default=0.0)
    if max_score <= 0:
        return []

    hits = [
        LexicalHit(chunk_id=chunk.id, score=score / max_score)
        for chunk, score in zip(corpus.chunks, scores, strict=True)
        if score > 0
    ]
    return sorted(hits, key=lambda hit: (-hit.score, hit.chunk_id))[:limit]


def chunk_tokens(chunk: ChunkRecord) -> list[str]:
    title = str(chunk.metadata.get("title") or "")
    anchor = str(chunk.metadata.get("anchor") or "")
    return tokenize(f"{title} {anchor} {chunk.text}")
