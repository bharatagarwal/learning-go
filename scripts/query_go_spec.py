#!/usr/bin/env python3
"""Retrieve deterministic Go spec grounding chunks for Codex answers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.go_spec_rag.config import DEFAULT_MANIFEST_PATH
from scripts.go_spec_rag.render import render_codex, render_json, render_markdown
from scripts.go_spec_rag.retrieval import query_index, status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Embed a question with the manifest's Ollama model and retrieve relevant "
            "Go spec chunks from ChromaDB."
        )
    )
    parser.add_argument("query", nargs="?", help="Question or search query.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--n-results", type=int, default=6)
    parser.add_argument(
        "--context-window",
        type=int,
        default=1,
        help="Include this many adjacent chunks on each side of each semantic hit.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["hybrid", "vector", "lexical", "cosine"],
        default="hybrid",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.0,
        help=(
            "Drop matches with cosine similarity below this floor. Used by --retrieval-mode cosine."
        ),
    )
    parser.add_argument("--semantic-candidates", type=int, default=32)
    parser.add_argument("--lexical-candidates", type=int, default=32)
    parser.add_argument("--parent-results", type=int, default=5)
    parser.add_argument("--max-parent-chars", type=int, default=5000)
    parser.add_argument(
        "--format",
        choices=["codex", "json", "markdown"],
        default="codex",
        help="Output format. 'codex' is a grounding packet intended for this agent.",
    )
    parser.add_argument("--status", action="store_true", help="Print index status JSON and exit.")
    parser.add_argument("--json", action="store_true", help="Alias for --format json.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    output_format = "json" if args.json else args.format

    try:
        if args.status:
            payload = status(args.manifest)
            print(render_json(payload), end="")
            return 0 if payload.get("ok", False) else 1

        if not args.query:
            parser.error("query is required unless --status is set")

        payload = query_index(
            args.query,
            manifest_path=args.manifest,
            n_results=args.n_results,
            context_window=args.context_window,
            retrieval_mode=args.retrieval_mode,
            semantic_candidates=args.semantic_candidates,
            lexical_candidates=args.lexical_candidates,
            parent_results=args.parent_results,
            max_parent_chars=args.max_parent_chars,
            similarity_threshold=args.similarity_threshold,
        )
        if output_format == "json":
            print(render_json(payload), end="")
        elif output_format == "markdown":
            print(render_markdown(payload), end="")
        else:
            print(render_codex(payload), end="")
        return 0
    except Exception as exc:
        print(render_json({"ok": False, "error": str(exc)}), file=sys.stderr, end="")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
