"""
A2A Messenger — inter-agent communication utilities.

Adapted from RDI-Foundation/agent-template. Provides helpers for sending
messages to other A2A agents and tracking conversation context.
"""

import json
import uuid
import logging
from typing import Optional

import httpx

log = logging.getLogger("agentbeats.messenger")


def create_message(
    text: str,
    context_id: Optional[str] = None,
    role: str = "user",
) -> dict:
    """Build an A2A message/send params payload."""
    msg = {
        "role": role,
        "parts": [{"kind": "text", "text": text}],
        "messageId": uuid.uuid4().hex,
    }
    if context_id:
        msg["contextId"] = context_id
    return {"message": msg}


def merge_parts(parts: list) -> str:
    """Merge a list of Part objects into a single string."""
    texts = []
    for part in parts:
        p = part.root if hasattr(part, "root") else part
        if hasattr(p, "text"):
            texts.append(p.text)
        elif hasattr(p, "data"):
            texts.append(json.dumps(p.data, default=str))
    return "\n".join(texts)


async def send_message(
    url: str,
    text: str,
    context_id: Optional[str] = None,
    timeout: float = 300,
) -> tuple[str, str]:
    """
    Send a message to a remote A2A agent and collect the response.

    Returns (response_text, context_id).
    """
    payload = create_message(text, context_id)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{url.rstrip('/')}/",
            json={
                "jsonrpc": "2.0",
                "method": "message/send",
                "id": uuid.uuid4().hex,
                "params": payload,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    result = data.get("result", {})
    if isinstance(result, dict) and "status" in result:
        text_out = ""
        for art in result.get("artifacts", []):
            for part in art.get("parts", []):
                if "text" in part:
                    text_out += part["text"]
                elif "data" in part:
                    text_out += json.dumps(part["data"], default=str)
        ctx = result.get("contextId", "")
        return text_out, ctx

    return str(result), ""


class Messenger:
    """Stateful A2A messenger that tracks conversation contexts per agent URL."""

    def __init__(self):
        self.contexts: dict[str, str] = {}

    async def talk_to_agent(
        self,
        url: str,
        text: str,
        reset: bool = False,
    ) -> str:
        """Send a message to another A2A agent, maintaining context."""
        if reset:
            self.contexts.pop(url, None)

        context_id = self.contexts.get(url)
        response_text, new_ctx = await send_message(url, text, context_id)
        if new_ctx:
            self.contexts[url] = new_ctx
        return response_text

    def reset(self):
        """Clear all conversation contexts."""
        self.contexts.clear()
