"""Best-effort redaction of endpoint and credential values in rendered logs."""

from __future__ import annotations

import re

_REDACTED = "<redacted>"
_SENSITIVE_NAME = r"(?:api_key|base_url|anthropic_api_key|anthropic_auth_token|ark_api_key)"

_QUOTED_VALUE_RE = re.compile(
    rf"(?P<prefix>[\"']?{_SENSITIVE_NAME}[\"']?\s*[:=]\s*)"
    rf"(?P<quote>[\"'])(?P<value>.*?)(?P=quote)",
    re.IGNORECASE,
)
_UNQUOTED_VALUE_RE = re.compile(
    rf"(?P<prefix>\b{_SENSITIVE_NAME}\b\s*[:=]\s*)"
    rf"(?P<value>(?![\"'])[^,\s}}\]\)]+)",
    re.IGNORECASE,
)
_ENDPOINT_URL_RE = re.compile(
    r"(?P<prefix>\bendpoint\s*[:=]\s*)(?P<quote>[\"']?)"
    r"(?P<value>https?://[^\s,\"'\)]+)(?P=quote)",
    re.IGNORECASE,
)
_BEARER_TOKEN_RE = re.compile(
    r"(?P<prefix>\bAuthorization\s*:\s*Bearer\s+)(?P<value>[^\s,]+)",
    re.IGNORECASE,
)
_RAW_API_KEY_RE = re.compile(r"\b(?:ark|sk-ant|sk)-[A-Za-z0-9_-]{24,}\b", re.IGNORECASE)


def _redact_sensitive_text(text: str) -> str:
    """Hide known endpoint and credential forms after a log record is rendered."""

    text = _QUOTED_VALUE_RE.sub(
        lambda match: f"{match.group('prefix')}{match.group('quote')}{_REDACTED}{match.group('quote')}",
        text,
    )
    text = _UNQUOTED_VALUE_RE.sub(lambda match: f"{match.group('prefix')}{_REDACTED}", text)
    text = _ENDPOINT_URL_RE.sub(
        lambda match: f"{match.group('prefix')}{match.group('quote')}{_REDACTED}{match.group('quote')}",
        text,
    )
    text = _BEARER_TOKEN_RE.sub(lambda match: f"{match.group('prefix')}{_REDACTED}", text)
    return _RAW_API_KEY_RE.sub(_REDACTED, text)
