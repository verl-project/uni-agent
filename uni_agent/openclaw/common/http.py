"""Tiny async JSON-over-HTTP helper shared by OpenClaw modules."""

from __future__ import annotations


async def post_json(
    url: str,
    payload: dict,
    *,
    headers: dict | None = None,
    timeout: float = 600.0,
) -> dict:
    """POST ``payload`` as JSON to ``url`` and return parsed JSON response."""
    import httpx

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        resp = await client.post(url, json=payload, headers=headers or {})
        resp.raise_for_status()
        return resp.json()
