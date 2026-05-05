from __future__ import annotations

from dataclasses import dataclass

import deal

from scripts.go_spec_rag.corpus import Corpus
from scripts.go_spec_rag.models import ParentRecord, SearchMatch


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
