"""Argument parsing tests for ``str_replace_editor``."""

import pytest
from pydantic import ValidationError

from uni_agent.tools.edit_file import EditFileTool, StrReplaceEditorArguments


def _parse_view_range(value):
    return StrReplaceEditorArguments(
        command="view",
        path="/testbed/example.py",
        view_range=value,
    ).view_range


@pytest.mark.parametrize("value", ([51, 63], "[51, 63]", "  [51, 63]\n"))
def test_view_range_accepts_list_and_json_encoded_list(value):
    assert _parse_view_range(value) == [51, 63]


@pytest.mark.parametrize(
    "value",
    (
        "not valid JSON",
        "51",
        '{"start": 51, "end": 63}',
        '["not-an-integer", 63]',
    ),
)
def test_view_range_rejects_invalid_json_shape_or_elements(value):
    with pytest.raises(ValidationError):
        _parse_view_range(value)


def test_view_range_schema_remains_an_integer_array():
    schema = EditFileTool(object()).schema()
    view_range_schema = schema["function"]["parameters"]["properties"]["view_range"]

    assert view_range_schema["type"] == "array"
    assert view_range_schema["items"] == {"type": "integer"}
