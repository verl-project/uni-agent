"""Schema normalization must distinguish schemas from names and literal data."""

import json
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, Field, create_model

from uni_agent.tools.base import Tool, Toolbox, ToolResult, build_function_schema

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


class _DocumentArguments(BaseModel):
    title: str
    subtitle: str | None = None


class _DocumentTool(Tool):
    name = "document"
    args_model = _DocumentArguments

    async def run(self, args, *, timeout=None):
        return ToolResult(text=json.dumps(args))


def test_title_argument_is_published_without_schema_title_metadata():
    parameters = _DocumentTool(object()).schema()["function"]["parameters"]

    assert "title" not in parameters
    assert parameters["properties"]["title"] == {"type": "string"}
    assert parameters["properties"]["subtitle"] == {"type": "string"}
    assert parameters["required"] == ["title"]


@pytest.mark.asyncio
async def test_toolbox_accepts_title_argument():
    result = await Toolbox([_DocumentTool(object())]).call("document", {"title": "Hello"})

    assert result.status == "ok"
    assert json.loads(result.text) == {"title": "Hello"}


@pytest.mark.parametrize("arguments", [{}, {"title": 42}, {"title": "Hello", "extra": True}])
@pytest.mark.asyncio
async def test_toolbox_still_rejects_invalid_document_arguments(arguments):
    result = await Toolbox([_DocumentTool(object())]).call("document", arguments)

    assert result.status == "format_error"


def test_nested_model_named_title_keeps_definition_and_properties():
    nested_model = create_model("title", title=(str, ...))
    arguments = create_model("Arguments", document=(nested_model, ...))
    parameters = build_function_schema("document", "", arguments)["function"]["parameters"]

    assert parameters["$defs"]["title"]["properties"]["title"] == {"type": "string"}
    assert "title" not in parameters["$defs"]["title"]
    validator = Draft202012Validator(parameters)
    validator.validate({"document": {"title": "Hello"}})
    assert not validator.is_valid({"document": {"title": 42}})


@pytest.mark.parametrize("keyword", ["default", "const", "enum", "examples"])
def test_literal_objects_are_not_normalized_as_schemas(keyword):
    literal: Any = {"title": "Hello", "default": None, "anyOf": [{"type": "null"}, {"type": "string"}]}
    if keyword in {"enum", "examples"}:
        literal = [literal]
    field = Field(default=literal) if keyword == "default" else Field(json_schema_extra={keyword: literal})
    arguments = create_model("Arguments", document=(dict[str, Any], field))
    parameters = build_function_schema("document", "", arguments)["function"]["parameters"]

    assert parameters["properties"]["document"][keyword] == literal
