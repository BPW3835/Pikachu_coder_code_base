"""Shared embedding client.

Isolated here so composed tools do not each rewire the same HTTP calls.
Production deployments typically point this at a managed embedding endpoint
(Cohere, Voyage, Anthropic's embeddings, or a self-hosted model behind
TEI / TGI). For the demo we hit an OpenAI-compatible endpoint.
"""

from __future__ import annotations

import os
from typing import Iterable

import httpx

from atlas_mcp.errors.framework import UpstreamError


class EmbeddingClient:
    """Tiny OpenAI-compatible embeddings client."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str = "text-embedding-3-small"):
        self.base_url = base_url or os.environ.get(
            "ATLAS_EMBEDDING_URL", "https://api.openai.com/v1"
        )
        self.api_key = api_key or os.environ.get("ATLAS_EMBEDDING_KEY", "")
        self.model = model

    async def embed(self, texts: Iterable[str]) -> list[list[float]]:
        body = {"model": self.model, "input": list(texts)}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.base_url.rstrip('/')}/embeddings", json=body, headers=headers
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise UpstreamError(
                "embedding_error",
                retryable=exc.response.status_code in (429, 500, 502, 503, 504),
                hint=str(exc),
                context={"status": exc.response.status_code},
            ) from exc
        except httpx.RequestError as exc:
            raise UpstreamError("embedding_network_error", retryable=True, hint=str(exc)) from exc
        return [row["embedding"] for row in data["data"]]
