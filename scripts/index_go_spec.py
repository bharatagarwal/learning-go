#!/usr/bin/env python3
"""Index the Go language specification into ChromaDB for local Codex RAG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.go_spec_rag.config import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION,
    DEFAULT_CORPUS_PATH,
    DEFAULT_DISTANCE_METRIC,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_SPEC_HTML,
)
from scripts.go_spec_rag.corpus import write_corpus
from scripts.go_spec_rag.indexing import (
    build_manifest,
    build_parent_records,
    chunk_sections,
    index_records,
)
from scripts.go_spec_rag.ollama import OllamaEmbeddingClient
from scripts.go_spec_rag.parse import parse_sections


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index the Go spec into ChromaDB using Chonkie + Ollama embeddings."
    )
    parser.add_argument("--spec-html", type=Path, default=DEFAULT_SPEC_HTML)
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--distance-metric", default=DEFAULT_DISTANCE_METRIC)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help=(
            "Character chunk size for Chonkie's RecursiveChunker. The default is "
            "conservative; bge-m3 supports up to ~8K tokens per input, so larger "
            "chunks are safe — measure the impact with eval_go_spec_retrieval.py."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not clear the collection first.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk only; do not embed.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.spec_html.exists():
        print(f"Spec HTML not found: {args.spec_html}", file=sys.stderr)
        return 2
    if args.chunk_size < 1:
        print("--chunk-size must be positive", file=sys.stderr)
        return 2
    if args.batch_size < 1:
        print("--batch-size must be positive", file=sys.stderr)
        return 2

    sections = parse_sections(args.spec_html)
    parents = build_parent_records(sections)
    records = chunk_sections(sections, chunk_size=args.chunk_size, source_file=args.spec_html)
    print(f"Parsed {len(sections)} sections into {len(records)} chunks.", file=sys.stderr)

    if args.dry_run:
        preview = {
            "sections": len(sections),
            "parents": len(parents),
            "chunks": len(records),
            "embedding_model": args.model,
            "first_chunk": records[0].__dict__ if records else None,
        }
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0

    embedder = OllamaEmbeddingClient(model=args.model, base_url=args.ollama_url)
    count, dimensions = index_records(
        records,
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        embedder=embedder,
        batch_size=args.batch_size,
        reset=not args.no_reset,
        distance_metric=args.distance_metric,
    )
    write_corpus(args.corpus_path, parents=parents, chunks=records)
    manifest = build_manifest(
        spec_html=args.spec_html,
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        corpus_path=args.corpus_path,
        model=args.model,
        ollama_url=args.ollama_url,
        chunk_size=args.chunk_size,
        parent_count=len(parents),
        section_count=len(sections),
        chunk_count=count,
        dimensions=dimensions,
        distance_metric=args.distance_metric,
    )
    manifest.write(args.manifest)
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
