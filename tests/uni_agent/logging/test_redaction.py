from __future__ import annotations

import logging

import pytest

from uni_agent.logging.handlers import _formatter
from uni_agent.logging.redaction import _redact_sensitive_text


@pytest.mark.parametrize(
    ("message", "secret"),
    [
        ('"base_url": "https://ark.example/api/compatible"', "https://ark.example/api/compatible"),
        ("'api_key': 'secret-value'", "secret-value"),
        ("BASE_URL=https://ark.example API_KEY=secret-value", "https://ark.example"),
        ("launch (endpoint=https://ark.example/api/compatible)", "https://ark.example/api/compatible"),
        ("Authorization: Bearer secret-token", "secret-token"),
        ("unexpected key ark-12345678-1234-1234-1234-123456789012", "ark-12345678-1234-1234-1234-123456789012"),
    ],
)
def test_redact_sensitive_text_hides_endpoint_and_credentials(message, secret):
    redacted = _redact_sensitive_text(message)

    assert secret not in redacted
    assert "<redacted>" in redacted


def test_redaction_preserves_non_secret_model_names():
    message = "model=ark-code-latest"
    assert _redact_sensitive_text(message) == message


def test_shared_formatter_redacts_rendered_log_arguments():
    record = logging.LogRecord(
        name="uni_agent.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="config=%s",
        args=({"base_url": "https://ark.example", "api_key": "secret-value"},),
        exc_info=None,
    )

    formatted = _formatter.format(record)

    assert "https://ark.example" not in formatted
    assert "secret-value" not in formatted
    assert formatted.count("<redacted>") == 2
