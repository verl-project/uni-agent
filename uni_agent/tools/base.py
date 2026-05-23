"""Abstract base class for scaffold tools."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel


class ToolFunctionSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolSchema(BaseModel):
    type: Literal["function"]
    function: ToolFunctionSchema


def _normalize_json_schema(value: Any) -> Any:
    """Normalize Pydantic JSON Schema for tool runtime usage."""
    if isinstance(value, list):
        return [_normalize_json_schema(item) for item in value]

    if not isinstance(value, dict):
        return value

    normalized = {}
    for key, item in value.items():
        if key == "title":
            continue

        normalized_item = _normalize_json_schema(item)
        if key == "default" and normalized_item is None:
            continue
        normalized[key] = normalized_item

    if "anyOf" in normalized:
        non_null_variants = [
            item
            for item in normalized["anyOf"]
            if not (isinstance(item, dict) and item.get("type") == "null" and len(item) == 1)
        ]
        if len(non_null_variants) == 1 and isinstance(non_null_variants[0], dict):
            merged = dict(non_null_variants[0])
            for key, item in normalized.items():
                if key != "anyOf":
                    merged[key] = item
            normalized = merged

    preferred_order = ("type", "description", "enum", "default", "items", "properties", "required")
    ordered = {}
    for key in preferred_order:
        if key in normalized:
            ordered[key] = normalized.pop(key)
    ordered.update(normalized)
    return ordered


class AbstractTool(ABC):
    """Abstract tool definition with description and install command."""

    copy_to_remote: bool = True
    """Whether ``install_tools`` should copy ``local_path`` into the runtime.

    True (default): framework-shipped tool scripts that get pushed to
    ``install_dir/<name>``. False: *system tools* whose binary the user
    installs separately and is already on PATH inside the runtime (e.g.
    ``lark-cli`` via ``npm install -g``, ``gh`` via apt). When False,
    ``install_tools`` skips copy+chmod and only runs ``get_install_command()``
    + ``which <name>`` as a presence check; ``local_path`` is ignored.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (e.g. execute_bash, str_replace_editor, submit)."""
        ...

    @property
    def local_path(self) -> Path | None:
        """Local script to copy into the runtime as ``install_dir/<name>``.
        Must point at an existing file when ``copy_to_remote`` is True;
        ignored (and may stay ``None``) otherwise.
        """
        return None

    @abstractmethod
    def get_tool_schema(self) -> dict:
        """
        OpenAI tool schema: { \"type\": \"function\", \"function\": { ... } }.
        """
        ...

    def build_tool_schema(self, description: str, arguments_model: type[BaseModel]) -> dict:
        """Build an OpenAI-compatible tool schema from a Pydantic arguments model."""
        parameters = _normalize_json_schema(arguments_model.model_json_schema())
        return ToolSchema(
            type="function",
            function=ToolFunctionSchema(
                name=self.name,
                description=description,
                parameters=parameters,
            ),
        ).model_dump()

    @abstractmethod
    def get_install_command(self) -> str | None:
        """Command to run in container to complete tool installation. Return None if no extra install step."""
        ...
