from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import chromadb
from chonkie import OverlapRefinery, RecursiveChunker

from scripts.go_spec_rag.config import (
    DEFAULT_CORPUS_PATH,
    DEFAULT_DISTANCE_METRIC,
    DEFAULT_QUERY_PREFIX,
    DEFAULT_SPEC_HTML,
    ROOT,
    SPEC_BASE_URL,
)
from scripts.go_spec_rag.models import (
    ChunkingConfig,
    ChunkRecord,
    EmbeddingConfig,
    IndexConfig,
    Manifest,
    Metadata,
    ParentRecord,
    Section,
    SourceConfig,
)
from scripts.go_spec_rag.ollama import OllamaEmbeddingClient
from scripts.go_spec_rag.pure import (
    clean_text,
    relative_display_path,
    sha256_file,
    stable_chunk_id,
    stable_section_id,
)


def chunk_sections(
    sections: Sequence[Section],
    *,
    chunk_size: int,
    chunk_overlap: int = 0,
    source_file: Path = DEFAULT_SPEC_HTML,
) -> list[ChunkRecord]:
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunker = RecursiveChunker(tokenizer="character", chunk_size=chunk_size)
    refinery = (
        OverlapRefinery(
            tokenizer="character",
            context_size=chunk_overlap,
            mode="recursive",
            method="prefix",
            merge=True,
        )
        if chunk_overlap > 0
        else None
    )
    records: list[ChunkRecord] = []

    for section_index, section in enumerate(sections):
        source_url = f"{SPEC_BASE_URL}#{section.anchor}" if section.anchor else SPEC_BASE_URL
        parent_id = stable_section_id(section_index, section.anchor, section.title)
        section_text = f"# {section.title}\n\n{section.text}"
        section_chunks = list(cast(Iterable[Any], chunker(section_text)))
        if refinery is not None and len(section_chunks) > 1:
            section_chunks = list(cast(Iterable[Any], refinery(section_chunks)))
        chunks = section_chunks

        for chunk_index, chunk in enumerate(chunks):
            text = clean_text(str(chunk.text))
            if not text:
                continue
            metadata: Metadata = {
                "source": "go_spec",
                "source_file": source_file.name,
                "url": source_url,
                "title": section.title,
                "anchor": section.anchor,
                "parent_id": parent_id,
                "heading_level": section.level,
                "section_index": section_index,
                "chunk_index": chunk_index,
                "char_count": len(text),
            }
            records.append(
                ChunkRecord(
                    id=stable_chunk_id(section_index, chunk_index, section.anchor, text),
                    text=text,
                    metadata=metadata,
                )
            )
    return records


def build_parent_records(sections: Sequence[Section]) -> list[ParentRecord]:
    parents: list[ParentRecord] = []
    for section_index, section in enumerate(sections):
        url = f"{SPEC_BASE_URL}#{section.anchor}" if section.anchor else SPEC_BASE_URL
        parents.append(
            ParentRecord(
                id=stable_section_id(section_index, section.anchor, section.title),
                title=section.title,
                anchor=section.anchor,
                level=section.level,
                text=f"# {section.title}\n\n{section.text}",
                url=url,
                section_index=section_index,
            )
        )
    return parents


def batched(items: Sequence[ChunkRecord], size: int) -> Iterator[Sequence[ChunkRecord]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def index_records(
    records: Sequence[ChunkRecord],
    *,
    chroma_path: Path,
    collection_name: str,
    embedder: OllamaEmbeddingClient,
    batch_size: int,
    reset: bool,
    distance_metric: str = DEFAULT_DISTANCE_METRIC,
) -> tuple[int, int]:
    collection = prepare_collection(
        chroma_path=chroma_path,
        collection_name=collection_name,
        embedder=embedder,
        reset=reset,
        distance_metric=distance_metric,
    )
    total = len(records)
    dimensions = 0
    for batch_number, batch in enumerate(batched(records, batch_size), start=1):
        print_batch_progress(batch_number, batch_size, total)
        embeddings = [embedder.embed(record.text) for record in batch]
        if embeddings and dimensions == 0:
            dimensions = len(embeddings[0])
        upsert_batch(collection, batch, embeddings)

    return collection.count(), dimensions


def prepare_collection(
    *,
    chroma_path: Path,
    collection_name: str,
    embedder: OllamaEmbeddingClient,
    reset: bool,
    distance_metric: str,
) -> Any:
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    existing_names = [collection.name for collection in client.list_collections()]
    if reset and collection_name in existing_names:
        client.delete_collection(collection_name)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={
            "source": "Go language specification",
            "source_url": SPEC_BASE_URL,
            "embedding_provider": "ollama",
            "embedding_model": embedder.model,
            "chunker": "chonkie.RecursiveChunker",
            "hnsw:space": distance_metric,
        },
    )


def print_batch_progress(batch_number: int, batch_size: int, total: int) -> None:
    start_index = (batch_number - 1) * batch_size + 1
    end_index = min(batch_number * batch_size, total)
    print(f"Embedding/indexing chunks {start_index}-{end_index} of {total}...", file=sys.stderr)


def upsert_batch(
    collection: Any, batch: Sequence[ChunkRecord], embeddings: list[list[float]]
) -> None:
    collection.upsert(
        ids=[record.id for record in batch],
        documents=[record.text for record in batch],
        metadatas=[record.metadata for record in batch],
        embeddings=cast(Any, embeddings),
    )


def build_manifest(
    *,
    spec_html: Path,
    chroma_path: Path,
    collection_name: str,
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    model: str,
    ollama_url: str,
    chunk_size: int,
    chunk_overlap: int,
    parent_count: int,
    section_count: int,
    chunk_count: int,
    dimensions: int,
    distance_metric: str = DEFAULT_DISTANCE_METRIC,
) -> Manifest:
    return Manifest(
        schema_version=3,
        source=SourceConfig(
            kind="go_spec_html",
            path=relative_display_path(spec_html, root=ROOT),
            sha256=sha256_file(spec_html),
            url=SPEC_BASE_URL,
        ),
        chunking=ChunkingConfig(
            library="chonkie",
            chunker="RecursiveChunker",
            tokenizer="character",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
        embedding=EmbeddingConfig(
            provider="ollama",
            model=model,
            base_url=ollama_url,
            dimensions=dimensions,
            query_prefix=DEFAULT_QUERY_PREFIX,
        ),
        index=IndexConfig(
            collection=collection_name,
            chroma_path=relative_display_path(chroma_path, root=ROOT),
            corpus_path=relative_display_path(corpus_path, root=ROOT),
            distance_metric=distance_metric,
            section_count=section_count,
            parent_count=parent_count,
            chunk_count=chunk_count,
        ),
    )
