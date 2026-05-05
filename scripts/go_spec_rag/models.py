from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

type JsonObject = dict[str, Any]
type MetadataValue = str | int | float | bool
type Metadata = dict[str, MetadataValue]


@dataclass(frozen=True)
class Section:
    title: str
    anchor: str
    level: str
    text: str


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    text: str
    metadata: Metadata

    def to_dict(self) -> JsonObject:
        return asdict(self)


@dataclass(frozen=True)
class ParentRecord:
    id: str
    title: str
    anchor: str
    level: str
    text: str
    url: str
    section_index: int

    def to_dict(self) -> JsonObject:
        return asdict(self)


@dataclass(frozen=True)
class SourceConfig:
    kind: str
    path: str
    sha256: str
    url: str


@dataclass(frozen=True)
class ChunkingConfig:
    library: str
    chunker: str
    tokenizer: str
    chunk_size: int
    chunk_overlap: int = 0


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    base_url: str
    dimensions: int
    query_prefix: str


@dataclass(frozen=True)
class IndexConfig:
    collection: str
    chroma_path: str
    corpus_path: str
    distance_metric: str
    section_count: int
    parent_count: int
    chunk_count: int


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    source: SourceConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    index: IndexConfig

    def to_dict(self) -> JsonObject:
        return asdict(self)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: JsonObject) -> Manifest:
        return cls(
            schema_version=int(data["schema_version"]),
            source=SourceConfig(**data["source"]),
            chunking=ChunkingConfig(**data["chunking"]),
            embedding=embedding_config_from_dict(data["embedding"]),
            index=index_config_from_dict(data["index"]),
        )


@dataclass(frozen=True)
class SearchMatch:
    rank: int
    id: str
    distance: float
    title: str
    anchor: str
    url: str
    chunk_index: int
    text: str
    parent_id: str = ""
    score: float = 0.0
    vector_score: float = 0.0
    lexical_score: float = 0.0
    sources: str = ""

    def to_dict(self) -> JsonObject:
        return asdict(self)


def embedding_config_from_dict(data: JsonObject) -> EmbeddingConfig:
    return EmbeddingConfig(
        provider=str(data["provider"]),
        model=str(data["model"]),
        base_url=str(data["base_url"]),
        dimensions=int(data["dimensions"]),
        query_prefix=str(data.get("query_prefix") or ""),
    )


def index_config_from_dict(data: JsonObject) -> IndexConfig:
    return IndexConfig(
        collection=str(data["collection"]),
        chroma_path=str(data["chroma_path"]),
        corpus_path=str(data.get("corpus_path") or ".rag/go_spec_corpus.json"),
        distance_metric=str(data.get("distance_metric") or "l2"),
        section_count=int(data["section_count"]),
        parent_count=int(data.get("parent_count") or data["section_count"]),
        chunk_count=int(data["chunk_count"]),
    )
