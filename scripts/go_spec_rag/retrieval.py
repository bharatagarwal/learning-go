from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import chromadb

from scripts.go_spec_rag.config import ROOT
from scripts.go_spec_rag.corpus import Corpus, read_corpus
from scripts.go_spec_rag.models import Manifest, SearchMatch
from scripts.go_spec_rag.ollama import OllamaEmbeddingClient
from scripts.go_spec_rag.pure import bounded_int, sha256_bytes, sha256_file
from scripts.go_spec_rag.rerank import parent_contexts


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
    n_results: int = 8,
    similarity_threshold: float = 0.0,
    max_parent_chars: int = 5000,
) -> dict[str, Any]:
    """Pure cosine top-K retrieval with optional similarity threshold.

    Embeds the query once with the manifest's recorded prefix, asks
    ChromaDB for top-K by cosine distance, drops matches with similarity
    below the threshold, and emits parent-section context for each
    surviving match.
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
        manifest_path=manifest_path,
        manifest=manifest,
        matches=matches,
        parents=parents,
        n_results=n_results,
        similarity_threshold=similarity_threshold,
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
    manifest_path: Path,
    manifest: Manifest,
    matches: list[SearchMatch],
    parents: list[Any],
    n_results: int,
    similarity_threshold: float,
) -> dict[str, Any]:
    return {
        "ok": True,
        "query": query,
        "embedded_query": manifest.embedding.query_prefix + query,
        "query_sha256": sha256_bytes(query.encode("utf-8")),
        "manifest_path": str(resolve_manifest_path(manifest_path)),
        "manifest_sha256": manifest_file_sha(manifest_path),
        "collection": manifest.index.collection,
        "embedding": manifest.embedding.__dict__,
        "chunking": manifest.chunking.__dict__,
        "index": manifest.index.__dict__,
        "source": manifest.source.__dict__,
        "retrieval": {
            "mode": "cosine",
            "requested_results": n_results,
            "similarity_threshold": similarity_threshold,
        },
        "returned_results": len(matches),
        "matches": [match.to_dict() for match in matches],
        "parent_contexts": [parent.to_dict() for parent in parents],
    }
