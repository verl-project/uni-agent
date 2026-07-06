"""
Quick connectivity test for the external OpenAI-compatible role LLM server.

Required environment variables:
  OPENAI_BASE_URL   - e.g. http://<cloud-ip>:30001/v1
  OPENAI_API_KEY    - API key accepted by the external server

Optional:
  EXTERNAL_MODEL    - model name (default: qwen3-4b-user-llm)

Usage:
  python test_vllm_server.py
"""

import os
import re
import sys

from openai import OpenAI


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>[\s\S]*?</think>\s*", "", text or "").strip()


def main():
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("EXTERNAL_MODEL", "qwen3-4b-user-llm").strip()

    if not base_url or not api_key:
        print("Error: set OPENAI_BASE_URL and OPENAI_API_KEY first.", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to: {base_url}")
    print(f"Model: {model}")
    print()

    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        models = client.models.list()
        print("Available models:")
        for m in models.data:
            print(f"  - {m.id}")
        print()
    except Exception as e:
        print(f"Failed to list models: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
            max_tokens=256,
        )
        raw = resp.choices[0].message.content
        print(f"Raw response:\n{raw}\n")
        print(f"Cleaned response:\n{strip_thinking(raw)}")
        print("\nConnection OK!")
    except Exception as e:
        print(f"Chat request failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
