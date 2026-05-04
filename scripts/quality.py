#!/usr/bin/env python3
"""Run uv-managed quality gates for the Go spec RAG scripts."""

from __future__ import annotations

import json

# This script is the controlled subprocess gate runner.
import subprocess  # nosec B404
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PYTHON_TARGETS = ["scripts", "tests"]


@dataclass(frozen=True)
class Gate:
    name: str
    command: Sequence[str]
    instruction: str
    uv_prefix: Sequence[str] = ("run",)


class GateFailure(Exception):
    def __init__(self, gate: Gate, returncode: int, violation: str) -> None:
        super().__init__(violation)
        self.gate = gate
        self.returncode = returncode
        self.violation = violation


def run(gate: Gate) -> None:
    uv_command = ["uv", *gate.uv_prefix, *gate.command]
    print("+ " + " ".join(uv_command), flush=True)
    try:
        # Commands come from static Gate definitions in this script.
        subprocess.run(uv_command, cwd=ROOT, check=True)  # nosec B603
    except subprocess.CalledProcessError as exc:
        raise GateFailure(
            gate=gate,
            returncode=exc.returncode,
            violation=f"Command exited with status {exc.returncode}.",
        ) from exc


def run_radon_gate() -> None:
    gate = Gate(
        name="radon-complexity",
        command=["radon", "cc", "--json", *PYTHON_TARGETS],
        instruction=(
            "Cyclomatic complexity exceeded. Extract nested conditionals or loops "
            "into separate private helper functions."
        ),
    )
    command = ["uv", "run", *gate.command]
    print("+ " + " ".join(command), flush=True)
    try:
        # Commands come from a static Gate definition in this script.
        completed = subprocess.run(  # nosec B603
            command,
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise GateFailure(
            gate=gate,
            returncode=exc.returncode,
            violation=f"Radon failed to analyze complexity; exit status {exc.returncode}.",
        ) from exc

    offenders = radon_offenders(json.loads(completed.stdout))
    if offenders:
        raise GateFailure(
            gate=gate,
            returncode=1,
            violation="Complexity rank exceeded B:\n" + "\n".join(offenders),
        )


def radon_offenders(report: dict[str, object]) -> list[str]:
    offenders: list[str] = []
    for filename, blocks in report.items():
        if not isinstance(blocks, list):
            continue
        offenders.extend(block_offenders(filename, blocks))
    return offenders


def block_offenders(filename: str, blocks: list[object]) -> list[str]:
    offenders: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        rank = str(block["rank"])
        if rank > "B":
            offenders.append(
                f"{filename}:{block['lineno']} {block['type']} "
                f"{block['name']} rank {rank} complexity {block['complexity']}"
            )
    return offenders


def print_gate_failure(error: GateFailure) -> None:
    print("", file=sys.stderr)
    print("QUALITY GATE FAILURE", file=sys.stderr)
    print(f"TOOL: {error.gate.name}", file=sys.stderr)
    print(f"VIOLATION: {error.violation}", file=sys.stderr)
    print(f"AGENT INSTRUCTION: {error.gate.instruction}", file=sys.stderr)
    print("", file=sys.stderr)


def main() -> int:
    gates = [
        Gate(
            "ruff-format",
            ["ruff", "format", "--check", *PYTHON_TARGETS],
            "Run `uv run ruff format scripts tests`, then inspect the formatted diff.",
        ),
        Gate(
            "ruff-lint",
            ["ruff", "check", *PYTHON_TARGETS],
            "Apply the reported lint fixes or run `uv run ruff check --fix scripts tests`.",
        ),
        Gate(
            "bandit-security",
            ["bandit", "-c", "pyproject.toml", "-r", "scripts"],
            (
                "Security scan failed. Remove risky calls, narrow inputs, or add a "
                "specific `# nosec` only with a local justification."
            ),
        ),
        Gate(
            "semgrep-security",
            ["semgrep", "--config", "p/default", "--error", "--metrics", "off", "scripts"],
            (
                "Semgrep found a risky pattern. Prefer safer APIs or add a narrow "
                "suppression only after documenting why the finding is false-positive."
            ),
            uv_prefix=("tool", "run"),
        ),
        Gate(
            "basedpyright",
            ["basedpyright", *PYTHON_TARGETS],
            "Fix the reported type errors; prefer narrowing types over `Any` casts.",
        ),
    ]
    try:
        for gate in gates:
            run(gate)
        run_radon_gate()
        run(
            Gate(
                "deal-lint",
                ["python", "-m", "deal", "lint", "scripts"],
                "Fix contract violations or remove stale contracts.",
            )
        )
        run(Gate("pytest", ["pytest"], "Fix the failing test or update the behavior contract."))
        run(
            Gate(
                "crosshair-pure",
                ["crosshair", "check", "scripts/go_spec_rag/pure.py"],
                "Fix the counterexample in the pure helper or strengthen its contract.",
            )
        )
        run(
            Gate(
                "crosshair-rerank",
                ["crosshair", "check", "scripts/go_spec_rag/rerank.py"],
                "Fix the counterexample in the rerank module or strengthen its contract.",
            )
        )
        run(
            Gate(
                "crosshair-lexical",
                ["crosshair", "check", "scripts/go_spec_rag/lexical.py"],
                "Fix the counterexample in the lexical module or strengthen its contract.",
            )
        )
    except GateFailure as exc:
        print_gate_failure(exc)
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
