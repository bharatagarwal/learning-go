# Methodology: A Local Loop For RAG Development

This document explains why the development setup in this repo is well-suited
to building retrieval-augmented generation systems, and what specifically a
fast local loop buys you.

## The RAG iteration problem

RAG systems are easy to stand up and hard to improve. The default workflow has
slow, expensive feedback at every layer:

- **Embeddings** cost money per call when you use a hosted provider, and
  re-embedding a corpus to try a new model is a budget decision.
- **Vector databases** in the cloud add network latency to every query and
  meter their writes.
- **Retrieval quality** is fuzzy. Did the change to chunking help or hurt? You
  can't tell from spot-checking a few queries.
- **Score fusion** combines vector, lexical, and rerank signals in ways that
  are hard to validate. Arbitrary weights ship and stay.
- **Edge cases** in scoring math (negative ranks, empty corpora, single-token
  queries) only surface in production.

Each of those frictions multiplies. By the time you've waited for embeddings,
paid for storage, and squinted at a handful of results, you've spent twenty
minutes confirming nothing.

## What "local loop" means here

Everything that gates a change runs on your laptop, in seconds:

| Concern | Local mechanism | Cost |
|---------|-----------------|------|
| Embeddings | Ollama with `mxbai-embed-large` | Free, ~50ms/query |
| Vector store | ChromaDB persistent client | Free, in-process |
| Lexical search | `rank_bm25` on the corpus JSON | Free, no network |
| Retrieval eval | `eval/go_spec_retrieval_cases.json` fixtures | Free, deterministic |
| Latency benchmark | `scripts/benchmark_retrieval.py` | Free, p95/p99 in seconds |
| Type/lint/security gates | `scripts/quality.py` | Free, ~30s |
| Property tests | Hypothesis via pytest | Free, generates inputs for you |
| Contract verification | CrossHair symbolic execution | Free, finds counterexamples |
| Format on save | PostToolUse hook → `ruff format` | Free, runs after each edit |

No cloud account, no API key, no rate limit. The longest single step is the
quality gate, and it's still under a minute.

## Why this matters specifically for RAG

### 1. Retrieval changes need measurement, not vibes

The RRF refactor in this repo replaced `0.54·vector + 0.26·lexical + 0.20·title`
with reciprocal rank fusion. That sounds principled until you ask: did it
actually help? The answer came from `scripts/eval_go_spec_retrieval.py`, which
measures expected-anchor recall and MRR over 22 curated cases. Anchor recall
went from "decent" to 100% on the fixtures.

Without an eval harness, the change is a guess. With one, it's a number.

### 2. Score fusion edge cases hide where you can't see them

`rrf_score` looks straightforward — sum of `1 / (k + rank)` across ranking
lists. But CrossHair, given the contract `result >= 0.0`, found a counterexample:
a negative rank produces a negative score. That's not theoretical; it would
have happened the first time someone passed an unsorted dict.

Symbolic execution doesn't replace tests, but it surfaces the cases you'd
never write tests for. For a system whose correctness depends on math, that's
a force multiplier.

### 3. Pure core, impure shell makes everything testable

`pure.py`, `rerank.py`, `render.py`, and most of `lexical.py` are pure
functions over data structures. The shell — `parse.py`, `corpus.py`,
`ollama.py`, `indexing.py`, `retrieval.py` — handles I/O.

This split means:
- The interesting logic (scoring, ranking, formatting) needs no mocks to test.
- Contracts (`@deal.pre`, `@deal.ensure`) work directly on the pure layer.
- The shell is thin enough that integration tests are cheap.

When you change scoring, you don't restart Ollama or rebuild the index. You
run pytest. It returns in seconds.

### 4. The manifest pins reproducibility

Every index writes `go_spec_manifest.json` recording the source file SHA-256,
chunk size, embedding model, dimension, and distance metric. Every query reads
the manifest and refuses to mismatch.

This sounds bureaucratic until you've spent an afternoon debugging a retrieval
regression that turned out to be a stale embedding model. Pinning the
contract makes that bug structurally impossible.

### 5. Hooks close the loop tighter

The `PostToolUse` hook runs `ruff format` after every Write/Edit. Formatting
is no longer a thing you remember to do; it happens before you notice. Same
principle applies if you wire up a pre-commit hook for the quality gate.

The point isn't formatting specifically. It's that small, automatable steps
should never wait on human attention.

## The development loop in practice

A typical change to retrieval scoring looks like this:

1. Edit `rerank.py` or `lexical.py`.
2. Save. Hook formats automatically.
3. `uv run pytest` — pure-function tests pass in <2 seconds.
4. `uv run python scripts/eval_go_spec_retrieval.py` — anchor recall and MRR
   over fixtures, ~10 seconds.
5. `uv run python scripts/benchmark_retrieval.py` — latency tradeoffs across
   modes, ~30 seconds.
6. `uv run python scripts/quality.py` — ruff, bandit, semgrep, basedpyright,
   radon, deal lint, pytest, CrossHair, ~60 seconds.
7. Commit if all green; revert if not.

Total wall-clock for a meaningful retrieval change: under two minutes. The
loop is short enough that you'll actually run it.

## What the local loop doesn't replace

It's not a substitute for production observability. The fixtures have 22
cases; production has unbounded query distributions. Latency on your laptop
is not latency on the user's device. Hosted embedding models drift over
versions in ways Ollama doesn't.

The local loop's job is to catch the regressions that *can* be caught
locally, fast, so the only things that reach production are the genuinely
hard ones.

## Why this generalizes beyond RAG

The pattern — pure functional core, contracts and property tests, eval
fixtures, local-only dependencies, hooks for the boring parts — works
anywhere the system has both:

- a math-flavored core (scoring, ranking, transforming) that benefits from
  symbolic verification, and
- an empirical question (does this change actually help?) that benefits from
  fixtures and benchmarks.

RAG happens to have both. So do search systems, recommenders, ML pipelines,
and most data tooling. The methodology isn't RAG-specific. It's just that RAG
makes the cost of *not* having a local loop most painful.
