# Codex Instructions For Go Spec RAG

When the user asks a question that should be grounded in the Go language specification,
run the local retrieval command before answering:

```bash
uv run python scripts/query_go_spec.py "<question>" --format codex --n-results 8
```

Answer only from the retrieved chunks. Cite the section title and URL for each factual
claim that depends on the spec. If the chunks do not contain enough evidence, say so
and either run a narrower follow-up retrieval or explain what is missing.

Retrieval is pure cosine top-K against ChromaDB. One query embedding, one nearest-
neighbor search, optional similarity floor via `--similarity-threshold`. No query
expansion, no lexical fusion, no per-parent diversification — the empirical eval
showed this matched a more complex hybrid pipeline while being simpler to reason
about.

The query command reads `.rag/go_spec_manifest.json` and embeds the question with the
same Ollama embedding model used to build the ChromaDB collection. Do not pass a
different embedding model at query time. Rebuild the index instead:

```bash
uv run python scripts/index_go_spec.py --model bge-m3
```

The indexer uses Chonkie's `RecursiveChunker` with a 1000-character default chunk
size and asks Ollama to embed with truncation disabled. If a future model rejects
that chunk size, rebuild with a smaller `--chunk-size` rather than allowing silent
truncation.

`bge-m3` is symmetric — queries and documents share the same embedding path,
so the manifest records an empty query prefix. If you switch to a model that
requires a query prefix (e.g. `mxbai-embed-large` wants
`Represent this sentence for searching relevant passages: `), rebuild the
index so the manifest captures it.

Use `uv` or `uvx` for all Python tooling in this repo. Do not use `pip`.
