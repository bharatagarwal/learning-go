from __future__ import annotations

from dataclasses import dataclass

import deal

from scripts.go_spec_rag.corpus import Corpus
from scripts.go_spec_rag.lexical import tokenize
from scripts.go_spec_rag.models import ChunkRecord, ParentRecord, SearchMatch

# RRF constant — standard value from Cormack et al. 2009
RRF_K = 60


@dataclass(frozen=True)
class CandidateScore:
    chunk_id: str
    rrf_score: float = 0.0
    vector_score: float = 0.0
    lexical_score: float = 0.0
    title_score: float = 0.0

    @property
    def score(self) -> float:
        return self.rrf_score

    @property
    def sources(self) -> str:
        labels: list[str] = []
        if self.vector_score > 0:
            labels.append("vector")
        if self.lexical_score > 0:
            labels.append("lexical")
        if self.title_score > 0:
            labels.append("title")
        return ",".join(labels)


@dataclass(frozen=True)
class ParentContext:
    rank: int
    parent_id: str
    title: str
    anchor: str
    url: str
    score: float
    text: str
    matched_child_ids: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "parent_id": self.parent_id,
            "title": self.title,
            "anchor": self.anchor,
            "url": self.url,
            "score": self.score,
            "text": self.text,
            "matched_child_ids": self.matched_child_ids,
        }


def merge_scores(
    *,
    corpus: Corpus,
    query: str,
    vector_scores: dict[str, float],
    lexical_scores: dict[str, float],
) -> list[CandidateScore]:
    """Merge vector and lexical scores using Reciprocal Rank Fusion (RRF).

    RRF formula: score = sum(1 / (k + rank)) across all ranking lists.
    This is more principled than arbitrary linear weights — it's what
    Elasticsearch and other production systems use for hybrid search.
    """
    chunks_by_id = corpus.chunks_by_id
    query_terms = set(tokenize(query))

    vector_ranks = scores_to_ranks(vector_scores)
    lexical_ranks = scores_to_ranks(lexical_scores)
    title_scores = {
        chunk_id: title_overlap_score(chunks_by_id[chunk_id], query_terms)
        for chunk_id in chunks_by_id
    }
    title_ranks = scores_to_ranks(title_scores)

    candidate_ids = set(vector_scores) | set(lexical_scores)
    candidates = [
        CandidateScore(
            chunk_id=chunk_id,
            rrf_score=rrf_score(chunk_id, vector_ranks, lexical_ranks, title_ranks),
            vector_score=vector_scores.get(chunk_id, 0.0),
            lexical_score=lexical_scores.get(chunk_id, 0.0),
            title_score=title_scores.get(chunk_id, 0.0),
        )
        for chunk_id in candidate_ids
        if chunk_id in chunks_by_id
    ]
    return sorted(candidates, key=lambda item: (-item.score, item.chunk_id))


@deal.ensure(lambda scores, result: len(result) == len(scores))
@deal.ensure(lambda scores, result: all(r >= 1 for r in result.values()))
def scores_to_ranks(scores: dict[str, float]) -> dict[str, int]:
    """Convert a score dict to ranks (1-indexed, higher score = lower rank number)."""
    sorted_ids = sorted(scores.keys(), key=lambda k: -scores[k])
    return {chunk_id: rank for rank, chunk_id in enumerate(sorted_ids, start=1)}


@deal.pre(
    lambda chunk_id, vector_ranks, lexical_ranks, title_ranks: all(
        r >= 1 for r in vector_ranks.values()
    )
)
@deal.pre(
    lambda chunk_id, vector_ranks, lexical_ranks, title_ranks: all(
        r >= 1 for r in lexical_ranks.values()
    )
)
@deal.pre(
    lambda chunk_id, vector_ranks, lexical_ranks, title_ranks: all(
        r >= 1 for r in title_ranks.values()
    )
)
@deal.ensure(lambda chunk_id, vector_ranks, lexical_ranks, title_ranks, result: result >= 0.0)
def rrf_score(
    chunk_id: str,
    vector_ranks: dict[str, int],
    lexical_ranks: dict[str, int],
    title_ranks: dict[str, int],
) -> float:
    """Compute RRF score: sum of 1/(k + rank) for each ranking list."""
    score = 0.0
    if chunk_id in vector_ranks:
        score += 1.0 / (RRF_K + vector_ranks[chunk_id])
    if chunk_id in lexical_ranks:
        score += 1.0 / (RRF_K + lexical_ranks[chunk_id])
    if chunk_id in title_ranks:
        score += 1.0 / (RRF_K + title_ranks[chunk_id])
    return score


def title_overlap_score(chunk: ChunkRecord, query_terms: set[str]) -> float:
    if not query_terms:
        return 0.0
    title_terms = set(tokenize(str(chunk.metadata.get("title") or "")))
    anchor_terms = set(tokenize(str(chunk.metadata.get("anchor") or "")))
    candidate_terms = title_terms | anchor_terms
    if not candidate_terms:
        return 0.0
    overlap = query_terms & candidate_terms
    query_coverage = len(overlap) / len(query_terms)
    title_coverage = len(overlap) / len(candidate_terms)
    return max(query_coverage, title_coverage)


def candidates_to_matches(corpus: Corpus, candidates: list[CandidateScore]) -> list[SearchMatch]:
    chunks_by_id = corpus.chunks_by_id
    matches: list[SearchMatch] = []
    for rank, candidate in enumerate(candidates, start=1):
        chunk = chunks_by_id[candidate.chunk_id]
        metadata = chunk.metadata
        matches.append(
            SearchMatch(
                rank=rank,
                id=chunk.id,
                distance=1.0 - candidate.vector_score,
                title=str(metadata.get("title") or "Untitled section"),
                anchor=str(metadata.get("anchor") or ""),
                url=str(metadata.get("url") or ""),
                chunk_index=int(metadata.get("chunk_index") or 0),
                text=chunk.text,
                parent_id=str(metadata.get("parent_id") or ""),
                score=candidate.score,
                vector_score=candidate.vector_score,
                lexical_score=candidate.lexical_score,
                sources=candidate.sources,
            )
        )
    return matches


def diversify_candidates_by_parent(
    corpus: Corpus,
    candidates: list[CandidateScore],
    *,
    limit: int,
    per_parent: int = 2,
) -> list[CandidateScore]:
    chunks_by_id = corpus.chunks_by_id
    selected: list[CandidateScore] = []
    parent_counts: dict[str, int] = {}

    for candidate in candidates:
        parent_id = str(chunks_by_id[candidate.chunk_id].metadata.get("parent_id") or "")
        if parent_counts.get(parent_id, 0) >= per_parent:
            continue
        selected.append(candidate)
        parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
        if len(selected) == limit:
            return selected

    for candidate in candidates:
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) == limit:
            return selected
    return selected


def parent_contexts(
    *,
    corpus: Corpus,
    matches: list[SearchMatch],
    limit: int,
    max_chars: int,
) -> list[ParentContext]:
    parent_scores: dict[str, float] = {}
    child_ids: dict[str, list[str]] = {}
    for match in matches:
        if not match.parent_id:
            continue
        parent_scores[match.parent_id] = max(parent_scores.get(match.parent_id, 0.0), match.score)
        child_ids.setdefault(match.parent_id, []).append(match.id)

    ranked_parent_ids = sorted(
        parent_scores, key=lambda parent_id: (-parent_scores[parent_id], parent_id)
    )
    contexts: list[ParentContext] = []
    for rank, parent_id in enumerate(ranked_parent_ids[:limit], start=1):
        parent = corpus.parents[parent_id]
        contexts.append(
            build_parent_context(rank, parent, parent_scores[parent_id], child_ids, max_chars)
        )
    return contexts


def build_parent_context(
    rank: int,
    parent: ParentRecord,
    score: float,
    child_ids: dict[str, list[str]],
    max_chars: int,
) -> ParentContext:
    return ParentContext(
        rank=rank,
        parent_id=parent.id,
        title=parent.title,
        anchor=parent.anchor,
        url=parent.url,
        score=score,
        text=truncate_parent_text(parent.text, max_chars),
        matched_child_ids=child_ids.get(parent.id, []),
    )


@deal.pre(lambda text, max_chars: max_chars >= 1 or max_chars < 1)  # accepts any max_chars
@deal.ensure(
    lambda text, max_chars, result: (
        max_chars < 1 or len(text) <= max_chars or len(result) <= max_chars + 15
    )
)
def truncate_parent_text(text: str, max_chars: int) -> str:
    if max_chars < 1 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"
