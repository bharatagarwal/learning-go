from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scripts.go_spec_rag.corpus import Corpus, read_corpus, write_corpus
from scripts.go_spec_rag.indexing import batched, chunk_sections
from scripts.go_spec_rag.lexical import lexical_search, query_variants, tokenize
from scripts.go_spec_rag.models import ChunkRecord, ParentRecord
from scripts.go_spec_rag.parse import parse_sections
from scripts.go_spec_rag.pure import bounded_int, clean_text, short_sha256, stable_chunk_id
from scripts.go_spec_rag.render import render, render_codex, render_json, render_markdown
from scripts.go_spec_rag.rerank import (
    candidates_to_matches,
    merge_scores,
    parent_contexts,
    rrf_score,
    scores_to_ranks,
    truncate_parent_text,
)


@given(st.text())
def test_clean_text_is_idempotent(text: str) -> None:
    assert clean_text(clean_text(text)) == clean_text(text)


@given(st.integers(), st.integers(min_value=1), st.integers(min_value=1))
def test_bounded_int_stays_inside_range(value: int, minimum: int, width: int) -> None:
    maximum = minimum + width
    result = bounded_int(value, minimum, maximum)
    assert minimum <= result <= maximum


def test_stable_chunk_id_is_repeatable() -> None:
    first = stable_chunk_id(3, 2, "Types", "type terms")
    second = stable_chunk_id(3, 2, "Types", "type terms")
    assert first == second
    assert first.startswith("go-spec-0003-002-")


def test_batched_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError, match="batch size"):
        list(batched([], 0))


def test_parse_sections_and_chunk_small_html(tmp_path: Path) -> None:
    spec = tmp_path / "spec.html"
    spec.write_text(
        """
        <html><body><article>
          <h1>The Go Programming Language Specification</h1>
          <h2 id="Types">Types</h2>
          <p>A type determines a set of values.</p>
          <h3 id="Boolean_types">Boolean types</h3>
          <p>A boolean type represents the set of Boolean truth values.</p>
        </article></body></html>
        """,
        encoding="utf-8",
    )

    sections = parse_sections(spec)
    assert [section.title for section in sections] == ["Types", "Boolean types"]
    records = chunk_sections(sections, chunk_size=200, source_file=spec)
    assert len(records) == 2
    assert str(records[0].metadata["url"]).endswith("#Types")
    assert "parent_id" in records[0].metadata


def test_chunk_sections_with_overlap_extends_chunks(tmp_path: Path) -> None:
    spec = tmp_path / "spec.html"
    paragraph_a = (
        "Slice types describe the relationship between a slice and its underlying array. " * 8
    )
    paragraph_b = (
        "Map types are unordered collections of key-value pairs. The key type must be comparable. "
        * 8
    )
    spec.write_text(
        f"""
        <html><body><article>
          <h2 id="Slice_types">Slice types</h2>
          <p>{paragraph_a}</p>
          <p>{paragraph_b}</p>
        </article></body></html>
        """,
        encoding="utf-8",
    )

    sections = parse_sections(spec)
    no_overlap = chunk_sections(sections, chunk_size=400, chunk_overlap=0, source_file=spec)
    with_overlap = chunk_sections(sections, chunk_size=400, chunk_overlap=80, source_file=spec)

    assert len(no_overlap) >= 2
    assert len(with_overlap) == len(no_overlap)
    longer_count = sum(
        len(b.text) > len(a.text) for a, b in zip(no_overlap, with_overlap, strict=True)
    )
    assert longer_count >= 1


def test_chunk_sections_rejects_overlap_at_or_above_chunk_size(tmp_path: Path) -> None:
    spec = tmp_path / "spec.html"
    spec.write_text(
        '<html><body><article><h2 id="X">X</h2><p>text</p></article></body></html>',
        encoding="utf-8",
    )
    sections = parse_sections(spec)
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_sections(sections, chunk_size=100, chunk_overlap=100, source_file=spec)
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_sections(sections, chunk_size=100, chunk_overlap=-1, source_file=spec)


def test_query_variants_add_spec_terms() -> None:
    variants = query_variants("Are slices comparable?")
    assert variants[0] == "Are slices comparable?"
    assert any("underlying array" in variant for variant in variants)
    assert any("comparison operators" in variant for variant in variants)
    assert "slice type underlying array length capacity nil make" in variants


def test_tokenize_preserves_identifier_like_terms() -> None:
    assert tokenize("len(s) and map[K]T") == ["len", "s", "and", "map", "k", "t"]


def test_lexical_search_ranks_expected_chunk(tmp_path: Path) -> None:
    parent = make_parent()
    slice_chunk = make_chunk("slice-1", parent, "Slice types", "Slices have len and cap.")
    map_chunk = make_chunk("map-1", parent, "Map types", "Map keys must be comparable.")
    corpus_path = tmp_path / "corpus.json"
    corpus = read_write_memory_corpus(corpus_path, [parent], [slice_chunk, map_chunk])

    hits = lexical_search(corpus, ["slice length capacity"], limit=2)

    assert hits[0].chunk_id == "slice-1"


def test_corpus_round_trip(tmp_path: Path) -> None:
    parent = make_parent()
    chunk = make_chunk("chunk-1", parent, "Slice types", "A slice has length.")
    path = tmp_path / "corpus.json"

    write_corpus(path, parents=[parent], chunks=[chunk])
    corpus = read_corpus(path)

    assert corpus.parents[parent.id].title == "Slice types"
    assert corpus.chunks_by_id[chunk.id].text == "A slice has length."


def test_rerank_merges_vector_and_lexical_scores(tmp_path: Path) -> None:
    parent = make_parent()
    slice_chunk = make_chunk("slice-1", parent, "Slice types", "Slices have len and cap.")
    map_chunk = make_chunk("map-1", parent, "Map types", "Map keys must be comparable.")
    corpus = read_write_memory_corpus(tmp_path / "corpus.json", [parent], [slice_chunk, map_chunk])

    candidates = merge_scores(
        corpus=corpus,
        query="slice capacity",
        vector_scores={"map-1": 0.4},
        lexical_scores={"slice-1": 1.0},
    )

    assert candidates[0].chunk_id == "slice-1"


def test_parent_contexts_return_ranked_parent_text(tmp_path: Path) -> None:
    parent = make_parent()
    chunk = make_chunk("slice-1", parent, "Slice types", "Slices have len and cap.")
    corpus = read_write_memory_corpus(tmp_path / "corpus.json", [parent], [chunk])
    candidates = merge_scores(
        corpus=corpus,
        query="slice",
        vector_scores={"slice-1": 0.8},
        lexical_scores={},
    )
    match = candidates_to_matches(corpus, candidates)[0]

    contexts = parent_contexts(corpus=corpus, matches=[match], limit=1, max_chars=80)

    assert contexts[0].title == "Slice types"
    assert contexts[0].matched_child_ids == ["slice-1"]


def make_parent() -> ParentRecord:
    return ParentRecord(
        id="parent-slice",
        title="Slice types",
        anchor="Slice_types",
        level="h3",
        text="# Slice types\n\nA slice describes an underlying array.",
        url="https://go.dev/ref/spec#Slice_types",
        section_index=1,
    )


def make_chunk(
    chunk_id: str,
    parent: ParentRecord,
    title: str,
    text: str,
) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id,
        text=text,
        metadata={
            "title": title,
            "anchor": parent.anchor,
            "parent_id": parent.id,
            "url": parent.url,
            "chunk_index": 0,
        },
    )


def read_write_memory_corpus(
    path: Path,
    parents: list[ParentRecord],
    chunks: list[ChunkRecord],
) -> Corpus:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_corpus(path, parents=parents, chunks=chunks)
    return read_corpus(path)


# --- RRF function tests ---


def test_scores_to_ranks_preserves_order() -> None:
    scores = {"a": 0.9, "b": 0.5, "c": 0.7}
    ranks = scores_to_ranks(scores)
    assert ranks["a"] == 1  # highest score = rank 1
    assert ranks["c"] == 2
    assert ranks["b"] == 3  # lowest score = rank 3


def test_scores_to_ranks_empty() -> None:
    assert scores_to_ranks({}) == {}


@given(st.dictionaries(st.text(min_size=1), st.floats(allow_nan=False, allow_infinity=False)))
def test_scores_to_ranks_length_preserved(scores: dict[str, float]) -> None:
    ranks = scores_to_ranks(scores)
    assert len(ranks) == len(scores)
    assert all(r >= 1 for r in ranks.values())


def test_rrf_score_single_list() -> None:
    vector_ranks = {"a": 1, "b": 2}
    score = rrf_score("a", vector_ranks, {}, {})
    assert score == pytest.approx(1.0 / 61)  # 1/(60+1)


def test_rrf_score_multiple_lists() -> None:
    vector_ranks = {"a": 1}
    lexical_ranks = {"a": 1}
    title_ranks = {"a": 1}
    score = rrf_score("a", vector_ranks, lexical_ranks, title_ranks)
    assert score == pytest.approx(3.0 / 61)  # 3 * 1/(60+1)


def test_rrf_score_missing_from_list() -> None:
    score = rrf_score("missing", {"a": 1}, {"b": 1}, {"c": 1})
    assert score == 0.0


# --- Render function tests ---


def test_render_json_produces_valid_json() -> None:
    payload = {"ok": True, "query": "test"}
    result = render_json(payload)
    import json

    parsed = json.loads(result)
    assert parsed["ok"] is True


def test_render_dispatches_to_correct_format() -> None:
    payload = make_minimal_payload()
    assert "GO SPEC RAG GROUNDING PACKET" in render(payload, "codex")
    assert "Go spec search results" in render(payload, "markdown")
    assert '"ok": true' in render(payload, "json")


def test_render_codex_includes_citation_instruction() -> None:
    payload = make_minimal_payload()
    result = render_codex(payload)
    assert "Cite section titles" in result


def test_render_markdown_includes_model_info() -> None:
    payload = make_minimal_payload()
    result = render_markdown(payload)
    assert "bge-m3" in result


# --- Truncate function tests ---


def test_truncate_parent_text_short_text_unchanged() -> None:
    assert truncate_parent_text("short", 100) == "short"


def test_truncate_parent_text_long_text_truncated() -> None:
    result = truncate_parent_text("a" * 200, 50)
    assert len(result) <= 65  # 50 + len("\n...[truncated]")
    assert result.endswith("[truncated]")


@given(st.text(), st.integers(min_value=1, max_value=1000))
def test_truncate_parent_text_never_exceeds_max_plus_suffix(text: str, max_chars: int) -> None:
    result = truncate_parent_text(text, max_chars)
    suffix_len = len("\n...[truncated]")
    assert len(result) <= max(len(text), max_chars + suffix_len)


# --- short_sha256 tests ---


def test_short_sha256_returns_correct_length() -> None:
    assert len(short_sha256("test", length=8)) == 8
    assert len(short_sha256("test", length=32)) == 32


def test_short_sha256_is_deterministic() -> None:
    assert short_sha256("hello") == short_sha256("hello")


@given(st.text(), st.integers(min_value=1, max_value=64))
def test_short_sha256_length_matches_request(text: str, length: int) -> None:
    assert len(short_sha256(text, length=length)) == length


def test_short_sha256_rejects_invalid_length() -> None:
    with pytest.raises(ValueError, match="length must be between"):
        short_sha256("test", length=0)
    with pytest.raises(ValueError, match="length must be between"):
        short_sha256("test", length=65)


# --- normalize_token & tokenize tests ---


def test_normalize_token_lowercases() -> None:
    from scripts.go_spec_rag.lexical import normalize_token

    assert normalize_token("Map") == "map"
    assert normalize_token("SLICE") == "slice"
    assert normalize_token("Type") == "type"


def test_normalize_token_plural() -> None:
    from scripts.go_spec_rag.lexical import normalize_token

    assert normalize_token("slices") == "slice"
    assert normalize_token("types") == "type"
    assert normalize_token("conversions") == "conversion"
    # -ies -> -y: "properties" -> "property"
    assert normalize_token("properties") == "property"
    # short token: just drop -s
    assert normalize_token("classes") == "classe"


def test_normalize_token_short_words_unchanged() -> None:
    from scripts.go_spec_rag.lexical import normalize_token

    assert normalize_token("map") == "map"
    assert normalize_token("nil") == "nil"
    assert normalize_token("len") == "len"
    assert normalize_token("cap") == "cap"


@given(st.from_regex(r"[A-Za-z_][A-Za-z0-9_]*", fullmatch=True))
def test_tokenize_round_trip_single_identifier(token: str) -> None:
    from scripts.go_spec_rag.lexical import normalize_token, tokenize

    result = tokenize(token)
    assert len(result) >= 1
    assert all(len(t) > 0 for t in result)
    # Re-tokenizing a normalized token should be idempotent
    normalized = normalize_token(token)
    re_result = tokenize(normalized)
    assert re_result == result


@given(st.lists(st.from_regex(r"[A-Za-z_][A-Za-z0-9_]*")))
def test_tokenize_always_produces_nonempty_tokens(tokens: list[str]) -> None:
    from scripts.go_spec_rag.lexical import tokenize

    if not tokens:
        return
    text = " ".join(tokens)
    result = tokenize(text)
    assert all(len(t) > 0 for t in result)


def test_tokenize_handles_punctuation() -> None:
    from scripts.go_spec_rag.lexical import tokenize

    assert tokenize("len(s) and map[K]T") == ["len", "s", "and", "map", "k", "t"]
    assert tokenize("a.b(c)") == ["a", "b", "c"]
    assert tokenize("ptr->field") == ["ptr", "field"]


# --- query_variants property tests ---


@given(st.text(min_size=1, max_size=100))
def test_query_variants_always_returns_at_least_one_variant(query: str) -> None:
    from scripts.go_spec_rag.lexical import query_variants

    variants = query_variants(query)
    assert len(variants) >= 1
    assert len(variants) <= 8


def test_query_variants_are_deterministic() -> None:
    from scripts.go_spec_rag.lexical import query_variants

    assert query_variants("Are slices comparable?") == query_variants("Are slices comparable?")


def test_query_variants_empty_query() -> None:
    from scripts.go_spec_rag.lexical import query_variants

    variants = query_variants("")
    assert len(variants) == 1
    assert variants[0] == ""


def test_query_variants_no_known_terms() -> None:
    from scripts.go_spec_rag.lexical import query_variants

    variants = query_variants("foo bar baz qux")
    assert variants[0] == "foo bar baz qux"
    assert len(variants) >= 1
    assert len(variants) <= 8


def test_query_variants_deduplicates() -> None:
    from scripts.go_spec_rag.lexical import query_variants

    # Repeated expansions should not produce duplicates
    variants = query_variants("map slice map")
    seen = set(variants)
    assert len(variants) == len(seen)


# --- Lexical search edge cases ---


def test_lexical_search_empty_variants(tmp_path: Path) -> None:
    parent = make_parent()
    chunk = make_chunk("c1", parent, "Slice types", "Slices have length.")
    corpus = read_write_memory_corpus(tmp_path / "corpus.json", [parent], [chunk])

    from scripts.go_spec_rag.lexical import lexical_search

    hits = lexical_search(corpus, [], limit=5)
    assert hits == []


def test_lexical_search_empty_corpus() -> None:
    from scripts.go_spec_rag.corpus import Corpus
    from scripts.go_spec_rag.lexical import lexical_search

    corpus = Corpus(parents={}, chunks=[])
    hits = lexical_search(corpus, ["slice"], limit=5)
    assert hits == []


def test_lexical_search_no_matches(tmp_path: Path) -> None:
    parent = make_parent()
    chunk = make_chunk("c1", parent, "Boolean types", "True and false.")
    corpus = read_write_memory_corpus(tmp_path / "corpus.json", [parent], [chunk])

    from scripts.go_spec_rag.lexical import lexical_search

    hits = lexical_search(corpus, ["xyzzy"], limit=5)
    assert hits == []


def test_lexical_search_rejects_non_positive_limit(tmp_path: Path) -> None:
    parent = make_parent()
    chunk = make_chunk("c1", parent, "Slice types", "Slices have length.")
    corpus = read_write_memory_corpus(tmp_path / "corpus.json", [parent], [chunk])

    from scripts.go_spec_rag.lexical import lexical_search

    with pytest.raises(ValueError, match="limit"):
        lexical_search(corpus, ["slice"], limit=0)


# --- HTML parsing edge cases ---


def test_parse_sections_multi_section_html(tmp_path: Path) -> None:
    spec = tmp_path / "spec.html"
    spec.write_text(
        """
        <html><body><article>
          <h2 id="Introduction">Introduction</h2>
          <p>This is the specification.</p>
          <h2 id="Notation">Notation</h2>
          <p>Syntax is described.</p>
          <h3 id="Syntax">Syntax</h3>
          <p>Using Extended Backus-Naur Form.</p>
          <h2 id="Source_code">Source code</h2>
          <p>Unicode text.</p>
          <h4 id="Characters">Characters</h4>
          <p>Unicode code points.</p>
          <pre>newline = 0x0A</pre>
          <h2 id="Constants">Constants</h2>
          <p>Constant values.</p>
        </article></body></html>
        """,
        encoding="utf-8",
    )

    sections = parse_sections(spec)
    titles = [section.title for section in sections]
    assert titles == [
        "Introduction",
        "Notation",
        "Syntax",
        "Source code",
        "Characters",
        "Constants",
    ]
    assert sections[4].anchor == "Characters"
    assert sections[4].level == "h4"
    assert "newline" in sections[4].text


def test_parse_sections_no_article_fallback(tmp_path: Path) -> None:
    """When there's no <article> tag, fall back to the whole soup.

    Headings at the top level of the soup are parsed; the classic <article>
    structure is preferred but the parser handles missing article elements.
    """
    spec = tmp_path / "spec.html"
    # Use an HTML fragment that BeautifulSoup won't wrap in <html><body>
    from bs4 import BeautifulSoup

    raw = """
        <h2 id="Introduction">Introduction</h2>
        <p>Body text.</p>
        <h2 id="Concepts">Concepts</h2>
        <p>More body text.</p>
    """
    soup = BeautifulSoup(raw, "lxml")
    spec.write_text(str(soup), encoding="utf-8")

    from scripts.go_spec_rag.parse import _select_article

    selected = _select_article(soup)
    # Should fall back to the soup itself when no <article> exists
    assert selected is soup


def test_parse_sections_removes_toc_nav(tmp_path: Path) -> None:
    """TOC, nav, and script elements should not appear in section text."""
    spec = tmp_path / "spec.html"
    spec.write_text(
        """
        <html><body><article>
          <nav id="nav">Navigation</nav>
          <div class="TOC">Table of Contents</div>
          <script>alert('xss')</script>
          <style>.hidden{}</style>
          <h2 id="Types">Types</h2>
          <p>A type determines a set of values.</p>
        </article></body></html>
        """,
        encoding="utf-8",
    )

    sections = parse_sections(spec)
    assert len(sections) == 1
    assert "Navigation" not in sections[0].text
    assert "Table of Contents" not in sections[0].text
    assert "alert" not in sections[0].text
    assert sections[0].title == "Types"


# --- Rerank diversification tests ---


def test_diversify_candidates_per_parent(tmp_path: Path) -> None:
    from scripts.go_spec_rag.rerank import diversify_candidates_by_parent, merge_scores

    parent = make_parent()
    chunks = [
        make_chunk(f"slice-{i}", parent, "Slice types", f"Slice variant {i}.") for i in range(5)
    ]
    corpus = read_write_memory_corpus(tmp_path / "corpus.json", [parent], chunks)

    candidates = merge_scores(
        corpus=corpus,
        query="slice",
        vector_scores={f"slice-{i}": 0.9 - i * 0.1 for i in range(5)},
        lexical_scores={},
    )

    diversified = diversify_candidates_by_parent(corpus, candidates, limit=8)
    # Only 2 per parent allowed before filler pass
    assert len(diversified) == 5
    assert diversified[0].chunk_id == "slice-0"
    assert diversified[1].chunk_id == "slice-1"


# --- Cross-module: chunk_sections preserves pre/code blocks ---


def test_chunk_sections_preserves_code_blocks(tmp_path: Path) -> None:
    spec = tmp_path / "spec.html"
    spec.write_text(
        """
        <html><body><article>
          <h2 id="Sample">Sample</h2>
          <p>Here is a code block:</p>
          <pre>
func main() {
    fmt.Println("hello")
}
          </pre>
          <p>That is the example.</p>
        </article></body></html>
        """,
        encoding="utf-8",
    )

    sections = parse_sections(spec)
    assert len(sections) == 1
    records = chunk_sections(sections, chunk_size=500, source_file=spec)
    assert len(records) == 1
    assert "func main()" in records[0].text
    assert "fmt.Println" in records[0].text


# --- Coverage for go_spec_rag.config constants ---


def test_config_constants_are_well_formed() -> None:
    from scripts.go_spec_rag.config import (
        DEFAULT_COLLECTION,
        DEFAULT_DISTANCE_METRIC,
        DEFAULT_MODEL,
        DEFAULT_QUERY_PREFIX,
        ROOT,
        SPEC_BASE_URL,
    )

    assert ROOT.exists()
    assert DEFAULT_COLLECTION == "go_spec"
    assert DEFAULT_DISTANCE_METRIC == "cosine"
    assert DEFAULT_MODEL == "bge-m3"
    assert DEFAULT_QUERY_PREFIX == ""
    assert SPEC_BASE_URL == "https://go.dev/ref/spec"


# --- Helpers ---


def make_minimal_payload() -> dict:
    return {
        "ok": True,
        "query": "test query",
        "query_sha256": "abc123",
        "manifest_sha256": "def456",
        "embedding": {"provider": "ollama", "model": "bge-m3", "query_prefix": ""},
        "retrieval": {"mode": "hybrid"},
        "matches": [],
        "context_chunks": [],
        "parent_contexts": [],
    }
