from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_HTML = (
    ROOT / "The Go Programming Language Specification - The Go Programming Language.html"
)
DEFAULT_CHROMA_PATH = ROOT / ".rag" / "chromadb"
DEFAULT_MANIFEST_PATH = ROOT / ".rag" / "go_spec_manifest.json"
DEFAULT_CORPUS_PATH = ROOT / ".rag" / "go_spec_corpus.json"
DEFAULT_COLLECTION = "go_spec"
DEFAULT_MODEL = "bge-m3"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_QUERY_PREFIX = ""
DEFAULT_DISTANCE_METRIC = "cosine"
SPEC_BASE_URL = "https://go.dev/ref/spec"
