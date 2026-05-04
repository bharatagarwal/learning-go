from __future__ import annotations

import http.client
import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class OllamaEmbeddingClient:
    model: str
    base_url: str
    retries: int = 3
    timeout_seconds: int = 120

    def embed(self, text: str) -> list[float]:
        payload = json.dumps(
            {
                "model": self.model,
                "input": text,
                "truncate": False,
            }
        ).encode("utf-8")

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                data = post_json(
                    self.base_url,
                    "/api/embed",
                    payload,
                    timeout_seconds=self.timeout_seconds,
                )
                return parse_embedding_response(data)
            except (TimeoutError, RuntimeError, json.JSONDecodeError, OSError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.5 * attempt)

        raise RuntimeError(
            f"Failed to embed text with Ollama model {self.model!r} after "
            f"{self.retries} attempts: {last_error}"
        )


def post_json(
    base_url: str,
    path: str,
    payload: bytes,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    parsed = urlparse(base_url)
    if parsed.scheme != "http":
        raise RuntimeError(f"Unsupported Ollama URL scheme: {parsed.scheme!r}")
    if not parsed.hostname:
        raise RuntimeError(f"Ollama URL is missing a hostname: {base_url!r}")

    target = normalized_target(parsed.path, path)
    connection = make_connection(
        scheme=parsed.scheme,
        host=parsed.hostname,
        port=parsed.port,
        timeout_seconds=timeout_seconds,
    )
    try:
        connection.request(
            "POST",
            target,
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status}: {body}")
        data = json.loads(body)
    finally:
        connection.close()

    if not isinstance(data, dict):
        raise RuntimeError(f"Ollama response was not a JSON object: {data!r}")
    return data


def normalized_target(base_path: str, endpoint_path: str) -> str:
    base = base_path.rstrip("/")
    endpoint = endpoint_path if endpoint_path.startswith("/") else f"/{endpoint_path}"
    return f"{base}{endpoint}" if base else endpoint


def make_connection(
    *,
    scheme: str,
    host: str,
    port: int | None,
    timeout_seconds: int,
) -> http.client.HTTPConnection:
    if scheme != "http":
        raise RuntimeError(f"Unsupported Ollama URL scheme: {scheme!r}")
    return http.client.HTTPConnection(host, port=port, timeout=timeout_seconds)


def parse_embedding_response(data: dict[str, Any]) -> list[float]:
    embeddings = data.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        first_embedding = embeddings[0]
        if isinstance(first_embedding, list) and first_embedding:
            return [float(value) for value in first_embedding]

    embedding = data.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError(f"Ollama response had no embedding: {data}")
    return [float(value) for value in embedding]
