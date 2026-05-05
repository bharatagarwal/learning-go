from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import chromadb

from scripts.go_spec_rag.config import ROOT, SPEC_BASE_URL
from scripts.go_spec_rag.corpus import Corpus, read_corpus
from scripts.go_spec_rag.lexical import lexical_search, query_variants
from scripts.go_spec_rag.models import Manifest, SearchMatch
from scripts.go_spec_rag.ollama import OllamaEmbeddingClient
from scripts.go_spec_rag.pure import bounded_int, sha256_bytes, sha256_file
from scripts.go_spec_rag.rerank import (
    candidates_to_matches,
    diversify_candidates_by_parent,
    merge_scores,
    parent_contexts,
)


def resolve_manifest_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_manifest(path: Path) -> Manifest:
    return Manifest.read(resolve_manifest_path(path))


def manifest_file_sha(path: Path) -> str:
    return sha256_file(resolve_manifest_path(path))


def get_collection(chroma_path: Path, collection_name: str):
    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        return client.get_collection(collection_name)
    except Exception as exc:  # Chroma raises version-specific exceptions.
        raise RuntimeError(
            f"Collection {collection_name!r} not found in {chroma_path}. "
            "Run: uv run python scripts/index_go_spec.py"
        ) from exc


def status(manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = load_manifest(manifest_path)
        collection = get_collection(
            resolve_repo_path(manifest.index.chroma_path),
            manifest.index.collection,
        )
        return {
            "ok": True,
            "manifest": str(resolve_manifest_path(manifest_path)),
            "manifest_sha256": manifest_file_sha(manifest_path),
            "collection": manifest.index.collection,
            "count": collection.count(),
            "embedding_model": manifest.embedding.model,
            "distance_metric": manifest.index.distance_metric,
            "chroma_path": manifest.index.chroma_path,
            "corpus_path": manifest.index.corpus_path,
        }
    except RuntimeError as exc:
        return {
            "ok": False,
            "manifest": str(resolve_manifest_path(manifest_path)),
            "count": 0,
            "error": str(exc),
        }


def query_index(
    query: str,
    *,
    manifest_path: Path,
    n_results: int,
    context_window: int,
    semantic_candidates: int = 24,
    lexical_candidates: int = 24,
    parent_results: int = 4,
    max_parent_chars: int = 5000,
    retrieval_mode: str = "hybrid",
    similarity_threshold: float = 0.0,
) -> dict[str, Any]:
    if retrieval_mode == "cosine":
        return cosine_query_index(
            query,
            manifest_path=manifest_path,
            n_results=n_results,
            similarity_threshold=similarity_threshold,
            max_parent_chars=max_parent_chars,
        )
    manifest = load_manifest(manifest_path)
    corpus = read_corpus(resolve_repo_path(manifest.index.corpus_path))
    collection = get_collection(
        resolve_repo_path(manifest.index.chroma_path),
        manifest.index.collection,
    )
    variants = query_variants(query)
    vector_scores = vector_candidate_scores(
        collection,
        variants,
        manifest=manifest,
        limit=bounded_int(semantic_candidates, 1, 80),
        enabled=retrieval_mode in {"hybrid", "vector"},
    )
    sparse_scores = lexical_candidate_scores(
        corpus,
        variants,
        limit=bounded_int(lexical_candidates, 1, 80),
        enabled=retrieval_mode in {"hybrid", "lexical"},
    )
    title_query = " ".join([query, *variants])
    matches = ranked_matches(corpus, title_query, vector_scores, sparse_scores, n_results)
    context_chunks = expand_context(
        corpus, matches, context_window=bounded_int(context_window, 0, 3)
    )
    parents = parent_contexts(
        corpus=corpus,
        matches=matches,
        limit=bounded_int(parent_results, 1, 10),
        max_chars=bounded_int(max_parent_chars, 500, 20000),
    )
    return retrieval_payload(
        query=query,
        variants=variants,
        manifest_path=manifest_path,
        manifest=manifest,
        retrieval_mode=retrieval_mode,
        matches=matches,
        context_chunks=context_chunks,
        parents=parents,
        n_results=n_results,
        context_window=context_window,
        semantic_candidates=semantic_candidates,
        lexical_candidates=lexical_candidates,
    )


def vector_candidate_scores(
    collection: Any,
    variants: Sequence[str],
    *,
    manifest: Manifest,
    limit: int,
    enabled: bool,
) -> dict[str, float]:
    if not enabled:
        return {}

    embedder = OllamaEmbeddingClient(
        model=manifest.embedding.model,
        base_url=manifest.embedding.base_url,
    )
    scores: dict[str, float] = {}
    for variant in variants:
        query_embedding = embedder.embed(manifest.embedding.query_prefix + variant)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["distances"],
        )
        merge_vector_scores(scores, cast(dict[str, Any], result))
    return scores


def merge_vector_scores(scores: dict[str, float], result: dict[str, Any]) -> None:
    ids = cast(list[str], result.get("ids", [[]])[0])
    distances = cast(list[float], result.get("distances", [[]])[0])
    for rank, (chunk_id, distance) in enumerate(zip(ids, distances, strict=False), start=1):
        cosine_score = max(0.0, 1.0 - float(distance))
        rank_score = 1.0 / rank
        score = (0.75 * cosine_score) + (0.25 * rank_score)
        scores[str(chunk_id)] = max(scores.get(str(chunk_id), 0.0), score)


def lexical_candidate_scores(
    corpus: Corpus,
    variants: Sequence[str],
    *,
    limit: int,
    enabled: bool,
) -> dict[str, float]:
    if not enabled:
        return {}
    return {hit.chunk_id: hit.score for hit in lexical_search(corpus, variants, limit=limit)}


def ranked_matches(
    corpus: Corpus,
    query: str,
    vector_scores: dict[str, float],
    lexical_scores: dict[str, float],
    n_results: int,
) -> list[SearchMatch]:
    candidates = merge_scores(
        corpus=corpus,
        query=query,
        vector_scores=vector_scores,
        lexical_scores=lexical_scores,
    )
    bounded_results = bounded_int(n_results, 1, 20)
    diversified = diversify_candidates_by_parent(corpus, candidates, limit=bounded_results)
    return candidates_to_matches(corpus, diversified)


def expand_context(
    corpus: Corpus,
    matches: list[SearchMatch],
    *,
    context_window: int,
) -> list[SearchMatch]:
    if context_window == 0:
        return matches

    by_key: dict[tuple[str, int], SearchMatch] = {}
    chunks_by_parent = group_chunks_by_parent(corpus)
    for match in matches:
        for neighbor in neighbors_for_match(chunks_by_parent, match, context_window):
            by_key[(neighbor.parent_id, neighbor.chunk_index)] = neighbor
    return sorted(
        by_key.values(),
        key=lambda item: (semantic_rank(matches, item), item.chunk_index, item.id),
    )


def group_chunks_by_parent(corpus: Corpus) -> dict[str, list[SearchMatch]]:
    grouped: dict[str, list[SearchMatch]] = {}
    for chunk in corpus.chunks:
        metadata = chunk.metadata
        parent_id = str(metadata.get("parent_id") or "")
        grouped.setdefault(parent_id, []).append(
            SearchMatch(
                rank=0,
                id=chunk.id,
                distance=1.0,
                title=str(metadata.get("title") or "Untitled section"),
                anchor=str(metadata.get("anchor") or ""),
                url=str(metadata.get("url") or SPEC_BASE_URL),
                chunk_index=int(metadata.get("chunk_index") or 0),
                text=chunk.text,
                parent_id=parent_id,
            )
        )
    for chunks in grouped.values():
        chunks.sort(key=lambda item: item.chunk_index)
    return grouped


def neighbors_for_match(
    chunks_by_parent: dict[str, list[SearchMatch]],
    match: SearchMatch,
    context_window: int,
) -> list[SearchMatch]:
    lower = match.chunk_index - context_window
    upper = match.chunk_index + context_window
    neighbors: list[SearchMatch] = []
    for neighbor in chunks_by_parent.get(match.parent_id, []):
        if lower <= neighbor.chunk_index <= upper:
            neighbors.append(copy_neighbor(match, neighbor))
    return neighbors


def copy_neighbor(match: SearchMatch, neighbor: SearchMatch) -> SearchMatch:
    return SearchMatch(
        rank=match.rank,
        id=neighbor.id,
        distance=match.distance,
        title=neighbor.title,
        anchor=neighbor.anchor,
        url=neighbor.url,
        chunk_index=neighbor.chunk_index,
        text=neighbor.text,
        parent_id=neighbor.parent_id,
        score=match.score,
        vector_score=match.vector_score,
        lexical_score=match.lexical_score,
        sources=match.sources,
    )


def semantic_rank(matches: list[SearchMatch], candidate: SearchMatch) -> int:
    for match in matches:
        if match.parent_id == candidate.parent_id:
            return match.rank
    return len(matches) + 1


def cosine_query_index(
    query: str,
    *,
    manifest_path: Path,
    n_results: int,
    similarity_threshold: float,
    max_parent_chars: int,
) -> dict[str, Any]:
    """Radically simple retrieval: pure cosine top-K with a similarity floor.

    Skips query-variant expansion, lexical search, RRF, per-parent
    diversification, and context-window expansion. The query is embedded
    once with the manifest's recorded prefix, ChromaDB returns top-K by
    cosine distance, and matches with similarity below the threshold are
    dropped. Parent contexts are still emitted so eval and downstream
    consumers see the same payload shape as hybrid mode.
    """
    manifest = load_manifest(manifest_path)
    corpus = read_corpus(resolve_repo_path(manifest.index.corpus_path))
    collection = get_collection(
        resolve_repo_path(manifest.index.chroma_path),
        manifest.index.collection,
    )
    embedder = OllamaEmbeddingClient(
        model=manifest.embedding.model,
        base_url=manifest.embedding.base_url,
    )
    query_embedding = embedder.embed(manifest.embedding.query_prefix + query)
    bounded_n = bounded_int(n_results, 1, 50)
    result = cast(
        dict[str, Any],
        collection.query(
            query_embeddings=[query_embedding],
            n_results=bounded_n,
            include=["distances"],
        ),
    )
    matches = cosine_matches(corpus, result, similarity_threshold)
    parents = parent_contexts(
        corpus=corpus,
        matches=matches,
        limit=max(1, len(matches)),
        max_chars=bounded_int(max_parent_chars, 500, 20000),
    )
    return retrieval_payload(
        query=query,
        variants=[query],
        manifest_path=manifest_path,
        manifest=manifest,
        retrieval_mode="cosine",
        matches=matches,
        context_chunks=matches,
        parents=parents,
        n_results=n_results,
        context_window=0,
        semantic_candidates=bounded_n,
        lexical_candidates=0,
    )


def cosine_matches(
    corpus: Corpus,
    chroma_result: dict[str, Any],
    similarity_threshold: float,
) -> list[SearchMatch]:
    ids = cast(list[str], chroma_result.get("ids", [[]])[0])
    distances = cast(list[float], chroma_result.get("distances", [[]])[0])
    chunks_by_id = corpus.chunks_by_id
    matches: list[SearchMatch] = []
    rank = 0
    for chunk_id, distance in zip(ids, distances, strict=False):
        cosine_score = max(0.0, 1.0 - float(distance))
        if cosine_score < similarity_threshold:
            continue
        chunk = chunks_by_id.get(str(chunk_id))
        if chunk is None:
            continue
        rank += 1
        metadata = chunk.metadata
        matches.append(
            SearchMatch(
                rank=rank,
                id=chunk.id,
                distance=float(distance),
                title=str(metadata.get("title") or "Untitled section"),
                anchor=str(metadata.get("anchor") or ""),
                url=str(metadata.get("url") or ""),
                chunk_index=int(metadata.get("chunk_index") or 0),
                text=chunk.text,
                parent_id=str(metadata.get("parent_id") or ""),
                score=cosine_score,
                vector_score=cosine_score,
                lexical_score=0.0,
                sources="vector",
            )
        )
    return matches


def retrieval_payload(
    *,
    query: str,
    variants: list[str],
    manifest_path: Path,
    manifest: Manifest,
    retrieval_mode: str,
    matches: list[SearchMatch],
    context_chunks: list[SearchMatch],
    parents: Sequence[Any],
    n_results: int,
    context_window: int,
    semantic_candidates: int,
    lexical_candidates: int,
) -> dict[str, Any]:
    return {
        "ok": True,
        "query": query,
        "query_variants": variants,
        "embedded_queries": [manifest.embedding.query_prefix + variant for variant in variants],
        "query_sha256": sha256_bytes(query.encode("utf-8")),
        "manifest_path": str(resolve_manifest_path(manifest_path)),
        "manifest_sha256": manifest_file_sha(manifest_path),
        "collection": manifest.index.collection,
        "embedding": manifest.embedding.__dict__,
        "chunking": manifest.chunking.__dict__,
        "index": manifest.index.__dict__,
        "source": manifest.source.__dict__,
        "retrieval": {
            "mode": retrieval_mode,
            "semantic_candidates": semantic_candidates,
            "lexical_candidates": lexical_candidates,
            "requested_results": n_results,
            "context_window": context_window,
        },
        "returned_results": len(matches),
        "matches": [match.to_dict() for match in matches],
        "context_chunks": [match.to_dict() for match in context_chunks],
        "parent_contexts": [parent.to_dict() for parent in parents],
    }
