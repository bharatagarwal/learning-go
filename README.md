# Go Spec RAG For Codex

This repo builds a local, deterministic RAG index over the saved Go language
specification HTML file.

## Setup

Use `uv` for all Python tooling:

```bash
uv sync --group dev
ollama pull mxbai-embed-large
```

Do not use `pip` in this repo.

## Build The Index

```bash
uv run python scripts/index_go_spec.py
```

The indexer:

- parses `The Go Programming Language Specification - The Go Programming Language.html`;
- chunks sections with Chonkie's `RecursiveChunker`;
- embeds chunks through local Ollama `mxbai-embed-large`;
- writes cosine vectors to `.rag/chromadb`;
- writes a parent/child corpus sidecar to `.rag/go_spec_corpus.json`;
- writes `.rag/go_spec_manifest.json`.

The manifest is the contract for deterministic querying. It records the source file
SHA-256, Chroma path, corpus path, collection name, distance metric, chunk size,
embedding model, embedding dimension, Ollama URL, and query prefix.

## Query For Grounding

```bash
uv run python scripts/query_go_spec.py "When is a Go type comparable?" --format codex
```

The query command always reads `.rag/go_spec_manifest.json`. It expands the query
deterministically, embeds those variants with the exact model and query prefix
recorded there, retrieves cosine nearest-neighbor chunks from Chroma, runs BM25-style
lexical retrieval over the corpus sidecar, merges both candidate pools, reranks with
title/anchor overlap, and returns parent sections plus child evidence.

Useful options:

- `--n-results 6`: number of semantic nearest-neighbor hits.
- `--context-window 1`: adjacent chunks to include around each hit.
- `--retrieval-mode hybrid`: choose `hybrid`, `vector`, or `lexical`.
- `--semantic-candidates 32` and `--lexical-candidates 32`: candidate pool sizes.
- `--parent-results 5`: number of parent sections returned for answering.
- `--format json`: machine-readable retrieval payload.
- `--status`: verify that the manifest and collection are present.

## Retrieval Evaluation

```bash
uv run python scripts/eval_go_spec_retrieval.py
```

The eval uses `eval/go_spec_retrieval_cases.json` (22 cases covering arrays,
slices, maps, channels, goroutines, defer, panic/recover, type inference,
interfaces, and more) and reports expected-anchor coverage and MRR. Add a case
whenever a retrieval failure or surprising answer appears.

Useful options:

- `--min-recall 0.8`: minimum anchor recall threshold (default: 0.8).
- `--retrieval-mode hybrid`: override the retrieval mode.
- `--json`: machine-readable output for CI pipelines.
- `--n-results 8` and `--context-window 1`: tune retrieval parameters.

## Benchmark

Measure retrieval latency across hybrid, vector, and lexical modes:

```bash
uv run python scripts/benchmark_retrieval.py
```

```bash
uv run python scripts/benchmark_retrieval.py --runs 20 --queries 4 --json
```

The benchmark runs a set of 10 queries against the built index, reports mean,
median, p95, and p99 latency in milliseconds. Add `--json` for machine-readable
output. Run `uv run python scripts/benchmark_retrieval.py --mode hybrid vector` to
compare retrieval strategies.

## Model Notes

`mxbai-embed-large` is the Ollama package for
`mixedbread-ai/mxbai-embed-large-v1`.

- Ollama lists it as a 335M embedding model with a 512-token context window:
  https://ollama.com/library/mxbai-embed-large
- The Hugging Face model card says retrieval queries should use the prefix
  `Represent this sentence for searching relevant passages: `, while documents do
  not need a prompt: https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
- Ollama's `/api/embed` endpoint supports `truncate`; this repo sets
  `truncate: false` so oversize chunks fail loudly:
  https://ollama.readthedocs.io/en/api/#generate-embeddings

Because the model has a short context window, the default chunk size is 1000
characters. If a future model rejects that size, rebuild with a smaller
`--chunk-size`.

## Quality Gate

```bash
uv run python scripts/quality.py
```

This runs Ruff formatting and linting, Bandit, Semgrep OSS default rules,
basedpyright, Ruff C901 complexity, Radon, deal lint, pytest with Hypothesis tests,
and CrossHair over `pure.py`, `rerank.py`, and `lexical.py`. Gate failures are
wrapped with an agent instruction that says how to fix the class of problem.

CrossHair performs symbolic execution against `@deal` contracts, finding
counterexamples by exploring edge cases in pure functions. If a contract is too weak
or has a bug, CrossHair will find it.

## Architecture

The retrieval pipeline is split into focused modules:

| Module | Responsibility |
|--------|----------------|
| `parse.py` | HTML → structured sections (BeautifulSoup) |
| `indexing.py` | Sections → chunks → ChromaDB embeddings |
| `lexical.py` | BM25L search, tokenization, query expansion |
| `rerank.py` | RRF fusion, per-parent diversification, parent assembly |
| `retrieval.py` | Orchestration: query → vector + lexical → rerank → format |
| `ollama.py` | Ollama HTTP embedding client with retry |
| `pure.py` | Pure functions with formal contracts (SHA-256, cleaning, etc.) |
| `render.py` | Output formatting (codex/markdown/json) |
| `corpus.py` | Corpus sidecar read/write |
| `models.py` | Strict dataclasses for all data types |
| `config.py` | All default paths, URLs, model constants |

Key design decisions:

- **Manifest as contract**: Every query checks `go_spec_manifest.json` so the
  retrieval is pinned to a specific source file, chunk size, and embedding model.
- **Hybrid retrieval with RRF**: Cosine vector search + BM25L, fused via
  Reciprocal Rank Fusion (Cormack et al. 2009, k=60).
- **Deterministic query expansion**: Hardcoded domain expansions, not LLM-generated
  synonyms.
- **Fail-fast**: `truncate: false` on embeddings, clear error messages with
  actionable commands for rebuilds.
- **Per-parent diversification**: At most 2 child chunks per parent section to
  keep answers broad.
- **Formal contracts**: `deal` pre/post-conditions on pure functions, checked at
  runtime and verified by CrossHair symbolic execution.
