"""Small provider boundary used by SGIRA executors.

Production code can wrap the repository's existing OpenAI-compatible client,
while tests use a deterministic fake. Keeping this boundary narrow makes the
four decision chains share exactly the same executor implementation.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol


class StructuredLLM(Protocol):
    def complete_json(
        self,
        *,
        role: str,
        system_prompt: str,
        payload: Mapping[str, Any],
    ) -> str | Mapping[str, Any]: ...


class ChatClientAdapter:
    """Adapt an object exposing ``chat(system_prompt, user_prompt)``."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def complete_json(
        self,
        *,
        role: str,
        system_prompt: str,
        payload: Mapping[str, Any],
    ) -> str:
        import json

        del role
        return self.client.chat(
            system_prompt,
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            temperature=0,
        )
