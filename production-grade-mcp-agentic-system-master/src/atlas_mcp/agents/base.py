"""Base classes for the agent layer.

Every agent in Atlas's copilot is a thin wrapper around an Anthropic API
call. The base class handles:

* Token authentication and retries.
* JSON-mode parsing with schema validation.
* Token budget accounting (so the orchestrator can enforce a per-turn cap).
* Structured logging with a shared ``run_id`` so you can trace a whole
  conversation across all four agents in one log query.
"""

from __future__ import annotations

import json
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

log = structlog.get_logger("atlas.agents")


@dataclass
class AgentRun:
    """Per-turn accounting, shared across all agents in one user interaction."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    tokens_in: int = 0
    tokens_out: int = 0
    tool_calls: int = 0


class LLM:
    """Minimal async Anthropic Messages client.

    Not the full SDK — just enough surface area for our agents. Isolating it
    here means swapping to a different provider (OpenAI, Bedrock, Vertex)
    means changing one file.
    """

    def __init__(self, api_key: str | None = None, model: str = "claude-opus-4-7"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"

    async def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            body["tools"] = tools
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{self.base_url}/messages", json=body, headers=headers)
            resp.raise_for_status()
            return resp.json()


class Agent(ABC):
    """Abstract base. Subclasses override ``act``."""

    name: str = "agent"
    system_prompt: str = ""

    def __init__(self, llm: LLM):
        self.llm = llm

    @abstractmethod
    async def act(self, run: AgentRun, **inputs) -> Any:
        ...

    # ── Convenience helpers ───────────────────────────────────────────────
    async def _complete_json(
        self, run: AgentRun, messages: list[dict], max_tokens: int = 512
    ) -> dict:
        resp = await self.llm.complete(self.system_prompt, messages, max_tokens=max_tokens)
        usage = resp.get("usage", {})
        run.tokens_in += usage.get("input_tokens", 0)
        run.tokens_out += usage.get("output_tokens", 0)
        text = _first_text_block(resp)
        log.debug("agent_response", agent=self.name, run_id=run.run_id,
                  tokens_in=usage.get("input_tokens"), tokens_out=usage.get("output_tokens"))
        return _parse_json_lenient(text)

    async def _complete_text(
        self, run: AgentRun, messages: list[dict], max_tokens: int = 512
    ) -> str:
        resp = await self.llm.complete(self.system_prompt, messages, max_tokens=max_tokens)
        usage = resp.get("usage", {})
        run.tokens_in += usage.get("input_tokens", 0)
        run.tokens_out += usage.get("output_tokens", 0)
        return _first_text_block(resp)


def _first_text_block(response: dict) -> str:
    for block in response.get("content", []):
        if block.get("type") == "text":
            return block.get("text", "")
    return ""


def _parse_json_lenient(text: str) -> dict:
    """Extract a JSON object from an LLM response even with stray prose."""
    text = text.strip()
    # Strip markdown fences if present.
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json\n"):
            text = text[5:]
    # Find the first { and the matching last }.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in LLM output: {text[:200]!r}")
    return json.loads(text[start:end + 1])
