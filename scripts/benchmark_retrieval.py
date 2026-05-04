#!/usr/bin/env python3
"""Benchmark retrieval latency for the Go spec RAG pipeline.

Runs a warmup query then measures wall-clock time, embedding time, and
total retrieval time across a fixed query set. Reports mean, median, p95,
and p99 latencies.

Usage:
    uv run python scripts/benchmark_retrieval.py
    uv run python scripts/benchmark_retrieval.py --runs 20 --queries 4
    uv run python scripts/benchmark_retrieval.py --json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.go_spec_rag.config import DEFAULT_MANIFEST_PATH
from scripts.go_spec_rag.retrieval import query_index

DEFAULT_QUERIES: list[str] = [
    "What is the difference between arrays and slices?",
    "Are slices comparable in Go?",
    "When is an array type comparable?",
    "What types can be map keys?",
    "How does panic and recover work in Go?",
    "What is included in a method set?",
    "When is a value assignable to a variable type?",
    "How does Go infer types from arguments?",
    "What does the predeclared identifier iota represent?",
    "How do you start a goroutine?",
]


@dataclass
class BenchmarkResult:
    label: str
    runs: int
    wall_ms: list[float] = field(default_factory=list)
    retrieval_mode: str = "hybrid"

    @property
    def count(self) -> int:
        return len(self.wall_ms)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.wall_ms) if self.wall_ms else 0.0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.wall_ms) if self.wall_ms else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.wall_ms) if self.wall_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.wall_ms) if self.wall_ms else 0.0

    @property
    def stdev_ms(self) -> float:
        return statistics.stdev(self.wall_ms) if len(self.wall_ms) >= 2 else 0.0

    @property
    def p95_ms(self) -> float:
        return percentile(self.wall_ms, 95) if self.wall_ms else 0.0

    @property
    def p99_ms(self) -> float:
        return percentile(self.wall_ms, 99) if self.wall_ms else 0.0

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "label": self.label,
            "retrieval_mode": self.retrieval_mode,
            "count": self.count,
            "mean_ms": round(self.mean_ms, 1),
            "median_ms": round(self.median_ms, 1),
            "min_ms": round(self.min_ms, 1),
            "max_ms": round(self.max_ms, 1),
            "stdev_ms": round(self.stdev_ms, 1),
            "p95_ms": round(self.p95_ms, 1),
            "p99_ms": round(self.p99_ms, 1),
        }


def percentile(data: list[float], p: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = max(0, min(len(sorted_data) - 1, round(len(sorted_data) * p / 100 - 0.5)))
    return sorted_data[index]


def warmup(manifest: Path, query: str = "What is a slice?") -> None:
    """One warmup query to ensure Ollama + Chroma are loaded into memory."""
    with contextlib.suppress(RuntimeError):
        query_index(
            query,
            manifest_path=manifest,
            n_results=3,
            context_window=0,
            retrieval_mode="hybrid",
        )


def measure(
    queries: list[str],
    *,
    manifest: Path,
    runs: int,
    retrieval_mode: str,
) -> BenchmarkResult:
    label = f"retrieval ({retrieval_mode}, {len(queries)} queries x {runs} runs)"
    result = BenchmarkResult(label=label, runs=runs, retrieval_mode=retrieval_mode)

    for _run_index in range(runs):
        for query in queries:
            try:
                start = time.perf_counter()
                query_index(
                    query,
                    manifest_path=manifest,
                    n_results=6,
                    context_window=1,
                    retrieval_mode=retrieval_mode,
                )
                elapsed = (time.perf_counter() - start) * 1000
                result.wall_ms.append(elapsed)
            except RuntimeError as exc:
                print(f"  [error] {exc}", file=sys.stderr)

    return result


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(c)) for w, c in zip(widths, row, strict=True)]
    separator = " | ".join("-" * w for w in widths)
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    print(f"  {header_line}")
    print(f"  {separator}")
    for row in rows:
        print("  " + " | ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)))
    print()


def print_human(results: list[BenchmarkResult]) -> None:
    for res in results:
        print(f"\n  [{res.label}]")
        print_table(
            ["metric", "value"],
            [
                ["queries", str(res.count)],
                ["mean", f"{res.mean_ms:.1f} ms"],
                ["median", f"{res.median_ms:.1f} ms"],
                ["min", f"{res.min_ms:.1f} ms"],
                ["max", f"{res.max_ms:.1f} ms"],
                ["stdev", f"{res.stdev_ms:.1f} ms"],
                ["p95", f"{res.p95_ms:.1f} ms"],
                ["p99", f"{res.p99_ms:.1f} ms"],
            ],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Go spec RAG retrieval latency.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of full passes over the query set (default: 5).",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=len(DEFAULT_QUERIES),
        help="Number of queries to use from the default set (default: all).",
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "vector", "lexical"],
        default="hybrid",
        help="Retrieval mode(s) to benchmark. Default runs all three.",
        nargs="+",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        print("Run 'uv run python scripts/index_go_spec.py' first.", file=sys.stderr)
        return 2

    queries = DEFAULT_QUERIES[: args.queries]
    print(
        f"Benchmarking {len(queries)} queries x {args.runs} runs modes={args.mode}...",
        file=sys.stderr,
    )

    # Warmup
    print("  Warmup...", file=sys.stderr)
    warmup(args.manifest)
    print("  Warmup complete.\n", file=sys.stderr)

    results: list[BenchmarkResult] = []
    for mode in args.mode:
        print(f"  Measuring mode={mode}...", file=sys.stderr)
        result = measure(
            queries,
            manifest=args.manifest,
            runs=args.runs,
            retrieval_mode=mode,
        )
        results.append(result)

    if args.json:
        payload = {
            "benchmark": {
                "query_count": len(queries),
                "runs": args.runs,
                "queries": queries,
            },
            "results": [r.to_dict() for r in results],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_human(results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
