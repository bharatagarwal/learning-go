"""Render retrieval payloads to various output formats."""

from __future__ import annotations

import json
from typing import Any, cast


def render(payload: dict[str, Any], output_format: str) -> str:
    """Render a retrieval payload to the specified format.

    Formats:
    - 'codex': Grounding packet for LLM agents with citation instructions
    - 'markdown': Human-readable search results
    - 'json': Machine-readable full payload
    """
    if output_format == "json":
        return render_json(payload)
    if output_format == "markdown":
        return render_markdown(payload)
    return render_codex(payload)


def render_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [f"Go spec search results for: {payload['query']!r}", ""]
    lines.append(f"Embedding model: {payload['embedding']['model']}")
    lines.append(f"Retrieval mode: {payload['retrieval']['mode']}")
    lines.append(f"Manifest: {payload['manifest_sha256']}")
    lines.append("")
    _append_parent_sections(lines, payload)
    _append_child_chunks(lines, payload)
    return "\n".join(lines).rstrip() + "\n"


def render_codex(payload: dict[str, Any]) -> str:
    lines = [
        "GO SPEC RAG GROUNDING PACKET",
        "",
        f"Question: {payload['query']}",
        f"Question SHA-256: {payload['query_sha256']}",
        f"Embedding model: {payload['embedding']['provider']}:{payload['embedding']['model']}",
        f"Query prefix: {payload['embedding']['query_prefix']!r}",
        f"Retrieval mode: {payload['retrieval']['mode']}",
        f"Manifest SHA-256: {payload['manifest_sha256']}",
        "",
        "Codex answer rule: answer only from these parent sections and child chunks. "
        "Cite section titles and URLs. If the chunks are insufficient, say what is missing.",
        "",
    ]
    _append_parent_sections(lines, payload)
    _append_child_chunks(lines, payload)
    return "\n".join(lines).rstrip() + "\n"


def _append_parent_sections(lines: list[str], payload: dict[str, Any]) -> None:
    parent_items = cast(list[dict[str, Any]], payload.get("parent_contexts") or [])
    if not parent_items:
        return
    lines.append("PARENT SECTIONS")
    lines.append("")
    for parent in parent_items:
        lines.append(f"[P{parent['rank']}] {parent['title']}")
        lines.append(f"URL: {parent['url']}")
        lines.append(f"Hybrid score: {float(parent['score']):.6f}")
        lines.append("Text:")
        lines.append(str(parent["text"]))
        lines.append("")


def _append_child_chunks(lines: list[str], payload: dict[str, Any]) -> None:
    chunk_items = cast(list[dict[str, Any]], payload.get("context_chunks") or payload["matches"])
    if not chunk_items:
        return
    lines.append("CHILD EVIDENCE")
    lines.append("")
    for match in chunk_items:
        lines.append(f"[C{match['rank']}.{match['chunk_index']}] {match['title']}")
        lines.append(f"URL: {match['url']}")
        lines.append(f"Chunk ID: {match['id']}")
        lines.append(
            "Scores: "
            f"hybrid={float(match['score']):.6f}, "
            f"vector={float(match['vector_score']):.6f}, "
            f"lexical={float(match['lexical_score']):.6f}, "
            f"sources={match['sources']}"
        )
        lines.append("Text:")
        lines.append(str(match["text"]))
        lines.append("")
