from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.go_spec_rag.models import ChunkRecord, Metadata, ParentRecord


@dataclass(frozen=True)
class Corpus:
    parents: dict[str, ParentRecord]
    chunks: list[ChunkRecord]

    @property
    def chunks_by_id(self) -> dict[str, ChunkRecord]:
        return {chunk.id: chunk for chunk in self.chunks}


def write_corpus(path: Path, *, parents: list[ParentRecord], chunks: list[ChunkRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "parents": [parent.to_dict() for parent in parents],
        "chunks": [chunk.to_dict() for chunk in chunks],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_corpus(path: Path) -> Corpus:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Corpus sidecar not found: {path}. Rebuild the index.") from exc

    parents = [parent_from_dict(item) for item in payload["parents"]]
    chunks = [chunk_from_dict(item) for item in payload["chunks"]]
    return Corpus(
        parents={parent.id: parent for parent in parents},
        chunks=chunks,
    )


def parent_from_dict(data: dict[str, Any]) -> ParentRecord:
    return ParentRecord(
        id=str(data["id"]),
        title=str(data["title"]),
        anchor=str(data["anchor"]),
        level=str(data["level"]),
        text=str(data["text"]),
        url=str(data["url"]),
        section_index=int(data["section_index"]),
    )


def chunk_from_dict(data: dict[str, Any]) -> ChunkRecord:
    raw_metadata = data["metadata"]
    if not isinstance(raw_metadata, dict):
        raise RuntimeError(f"Invalid chunk metadata for {data.get('id')}")
    metadata: Metadata = {}
    for key, value in raw_metadata.items():
        if isinstance(value, str | int | float | bool):
            metadata[str(key)] = value
    return ChunkRecord(
        id=str(data["id"]),
        text=str(data["text"]),
        metadata=metadata,
    )
