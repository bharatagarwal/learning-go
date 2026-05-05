#!/usr/bin/env python3
"""Evaluate Go spec retrieval against a small labeled anchor set."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.go_spec_rag.config import DEFAULT_MANIFEST_PATH, ROOT
from scripts.go_spec_rag.render import render_json
from scripts.go_spec_rag.retrieval import query_index

DEFAULT_CASES = ROOT / "eval" / "go_spec_retrieval_cases.json"


@dataclass(frozen=True)
class EvalCase:
    id: str
    query: str
    expected_anchors: list[str]


def load_cases(path: Path) -> list[EvalCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalCase(
            id=str(item["id"]),
            query=str(item["query"]),
            expected_anchors=[str(anchor) for anchor in item["expected_anchors"]],
        )
        for item in data
    ]


def evaluate_case(case: EvalCase, args: argparse.Namespace) -> dict[str, Any]:
    payload = query_index(
        case.query,
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
    parent_anchors = [str(parent["anchor"]) for parent in payload["parent_contexts"]]
    child_anchors = [str(match["anchor"]) for match in payload["matches"]]
    ranks = ranks_for_expected(parent_anchors, case.expected_anchors)
    covered = sorted(set(parent_anchors) & set(case.expected_anchors))
    anchor_recall = len(covered) / len(case.expected_anchors)
    return {
        "id": case.id,
        "query": case.query,
        "expected_anchors": case.expected_anchors,
        "covered_anchors": covered,
        "parent_anchors": parent_anchors,
        "child_anchors": child_anchors,
        "hit": anchor_recall == 1.0,
        "anchor_recall": anchor_recall,
        "first_rank": min(ranks) if ranks else None,
        "mrr": 1 / min(ranks) if ranks else 0.0,
    }


def ranks_for_expected(retrieved: list[str], expected: list[str]) -> list[int]:
    expected_set = set(expected)
    return [index for index, anchor in enumerate(retrieved, start=1) if anchor in expected_set]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    hits = sum(1 for result in results if result["hit"])
    mrr = sum(float(result["mrr"]) for result in results) / count if count else 0.0
    anchor_recall = (
        sum(float(result["anchor_recall"]) for result in results) / count if count else 0.0
    )
    return {
        "case_count": count,
        "full_coverage_hits": hits,
        "full_coverage_rate": hits / count if count else 0.0,
        "anchor_recall_at_parent_k": anchor_recall,
        "mrr": mrr,
        "failures": [result for result in results if not result["hit"]],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Go spec RAG retrieval.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
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
    parser.add_argument("--n-results", type=int, default=8)
    parser.add_argument("--context-window", type=int, default=1)
    parser.add_argument("--semantic-candidates", type=int, default=32)
    parser.add_argument("--lexical-candidates", type=int, default=32)
    parser.add_argument("--parent-results", type=int, default=5)
    parser.add_argument("--max-parent-chars", type=int, default=3500)
    parser.add_argument("--min-recall", type=float, default=0.8)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cases = load_cases(args.cases)
    results = [evaluate_case(case, args) for case in cases]
    summary = summarize(results)
    payload = {
        "ok": summary["anchor_recall_at_parent_k"] >= args.min_recall,
        "summary": summary,
    }
    if args.json:
        print(render_json(payload), end="")
    else:
        print_human_summary(payload)
    return 0 if payload["ok"] else 1


def print_human_summary(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print(
        "Go spec retrieval eval: "
        f"anchor_recall={summary['anchor_recall_at_parent_k']:.3f}, "
        f"mrr={summary['mrr']:.3f}, "
        f"full_coverage={summary['full_coverage_hits']}/{summary['case_count']}"
    )
    failures = summary["failures"]
    if failures:
        print("Failures:")
        for failure in failures:
            print(
                f"- {failure['id']}: expected {failure['expected_anchors']}, "
                f"got {failure['parent_anchors']}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
