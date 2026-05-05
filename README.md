# Go Spec RAG For Codex

This repo builds a local, deterministic RAG index over the saved Go language
specification HTML file.

## Setup

Use `uv` for all Python tooling:

```bash
uv sync --group dev
ollama pull bge-m3
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

The query command always reads `.rag/go_spec_manifest.json`. It embeds the query
once with the model and query prefix recorded there, asks ChromaDB for top-K
nearest neighbors by cosine distance, optionally drops matches below a similarity
threshold, and returns parent sections with child evidence chunks.

Useful options:

- `--n-results 8`: number of cosine nearest-neighbor hits returned (default: 8).
- `--similarity-threshold 0.6`: drop matches with cosine similarity below this floor.
- `--max-parent-chars 5000`: per-parent text limit before truncation.
- `--format json|markdown|codex`: output format. `codex` is a grounding packet for agents.
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
- `--n-results 8`: tune top-K returned per query.
- `--similarity-threshold 0.5`: tighten/loosen the cosine floor.
- `--json`: machine-readable output for CI pipelines.

## Benchmark

```bash
uv run python scripts/benchmark_retrieval.py
uv run python scripts/benchmark_retrieval.py --runs 20 --queries 4 --json
```

The benchmark runs the default 10-query set against the built index and reports
mean, median, p95, and p99 latency in milliseconds.

## Model Notes

`bge-m3` is the Ollama package for `BAAI/bge-m3`.

- Ollama lists it as a 567M embedding model with an 8192-token context window:
  https://ollama.com/library/bge-m3
- The Hugging Face model card states bge-m3 is symmetric — queries and
  documents use the same embedding path, no query prefix required:
  https://huggingface.co/BAAI/bge-m3
- Strong multilingual support across 100+ languages, with documented Hindi
  retrieval performance (Hindi-BEIR 47.29 nDCG@10).
- Ollama's `/api/embed` endpoint supports `truncate`; this repo sets
  `truncate: false` so oversize chunks fail loudly:
  https://ollama.readthedocs.io/en/api/#generate-embeddings

The default chunk size is a conservative 1000 characters. bge-m3's 8K-token
context window makes much larger chunks safe — experiment with
`--chunk-size 4000` and measure with `eval_go_spec_retrieval.py` before
committing to a new default.

## Quality Gate

```bash
uv run python scripts/quality.py
```

This runs Ruff formatting and linting, Bandit, Semgrep OSS default rules,
basedpyright, Ruff C901 complexity, Radon, deal lint, pytest with Hypothesis tests,
and CrossHair over `pure.py` and `rerank.py`. Gate failures are wrapped with an
agent instruction that says how to fix the class of problem.

CrossHair performs symbolic execution against `@deal` contracts, finding
counterexamples by exploring edge cases in pure functions. If a contract is too weak
or has a bug, CrossHair will find it.

## Architecture

The retrieval pipeline is split into focused modules:

| Module | Responsibility |
|--------|----------------|
| `parse.py` | HTML → structured sections (BeautifulSoup) |
| `indexing.py` | Sections → chunks → ChromaDB embeddings |
| `retrieval.py` | Cosine top-K query against ChromaDB with optional threshold |
| `rerank.py` | Parent-section assembly and text truncation |
| `ollama.py` | Ollama HTTP embedding client with retry |
| `pure.py` | Pure functions with formal contracts (SHA-256, cleaning, etc.) |
| `render.py` | Output formatting (codex/markdown/json) |
| `corpus.py` | Corpus sidecar read/write |
| `models.py` | Strict dataclasses for all data types |
| `config.py` | All default paths, URLs, model constants |

Key design decisions:

- **Manifest as contract**: Every query checks `go_spec_manifest.json` so the
  retrieval is pinned to a specific source file, chunk size, and embedding model.
- **Pure cosine top-K**: One query embedding, ChromaDB nearest-neighbor search,
  optional similarity floor. No query expansion, no lexical fusion, no rerank
  weights — empirically it matched a hybrid pipeline on this corpus while being
  far simpler.
- **Fail-fast**: `truncate: false` on embeddings, clear error messages with
  actionable commands for rebuilds.
- **Formal contracts**: `deal` pre/post-conditions on pure functions, checked at
  runtime and verified by CrossHair symbolic execution.
