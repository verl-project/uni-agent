import ast
import json
import uuid
from typing import Any

import regex

from uni_agent.interaction.tool_schemas import (
    OpenAIFunctionCallSchema,
    OpenAIFunctionToolCall,
    OpenAIFunctionToolSchema,
)


class FunctionCallFormatError(Exception):
    pass


# modified from qwen3 coder tool parser
class XMLToolParser:
    def __init__(self):
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_prefix: str = "<function="

        # Regex patterns
        self.tool_call_complete_regex = regex.compile(r"<tool_call>(.*?)</tool_call>", regex.DOTALL)
        self.tool_call_regex = regex.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", regex.DOTALL)
        self.tool_call_function_regex = regex.compile(r"<function=(.*?)</function>|<function=(.*)$", regex.DOTALL)
        self.tool_call_parameter_regex = regex.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", regex.DOTALL
        )

    def _get_arguments_config(self, func_name: str, tools: list[OpenAIFunctionToolSchema]) -> dict:
        for config in tools:
            assert config.type == "function"
            if config.function.name == func_name:
                properties = config.function.parameters.properties
                return {k: v.model_dump() for k, v in properties.items()}
        raise FunctionCallFormatError(
            f"Invalid action: function '{func_name}' is not defined in the tools list.\n"
            f"Allowed functions should be one of: {[tool.function.name for tool in tools]}."
        )

    def _convert_param_value(self, param_value: str, param_name: str, param_config: dict, func_name: str) -> Any:
        """Convert parameter value based on its type in the schema."""
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                raise FunctionCallFormatError(
                    f"Invalid action: parameter '{param_name}' is not defined "
                    f"in the parameters for function '{func_name}'.\n"
                    f"Allowed parameters for function '{func_name}': {list(param_config.keys())}."
                )
            return param_value

        if isinstance(param_config[param_name], dict) and "type" in param_config[param_name]:
            param_type = str(param_config[param_name]["type"]).strip().lower()
        else:
            param_type = "string"
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                param_value = int(param_value)
                return param_value
            except Exception:
                raise FunctionCallFormatError(
                    f"Invalid action: value '{param_value}' of parameter '{param_name}' "
                    f"is not an integer in tool call '{func_name}'."
                ) from None
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                float_param_value = float(param_value)
                param_value = (
                    float_param_value if float_param_value - int(float_param_value) != 0 else int(float_param_value)
                )
                return param_value
            except Exception:
                raise FunctionCallFormatError(
                    f"Invalid action: value '{param_value}' of parameter '{param_name}' "
                    f"is not a float in tool call '{func_name}'."
                ) from None
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value in ["true", "false"]:
                return param_value == "true"
            raise FunctionCallFormatError(
                f"Invalid action: value '{param_value}' of parameter '{param_name}' "
                f"is not a boolean (`true` of `false`) in tool call '{func_name}'."
            )
        else:
            if (
                param_type in ["object", "array", "arr"]
                or param_type.startswith("dict")
                or param_type.startswith("list")
            ):
                try:
                    param_value = json.loads(param_value)
                    return param_value
                except Exception:
                    pass
            try:
                param_value = ast.literal_eval(param_value)  # safer
                return param_value
            except Exception:
                raise FunctionCallFormatError(
                    f"Invalid action: value '{param_value}' of parameter '{param_name}' "
                    f"is not valid in tool call '{func_name}'."
                ) from None

    def _parse_xml_function_call(
        self, function_call_str: str, tools: list[OpenAIFunctionToolSchema]
    ) -> OpenAIFunctionToolCall:
        # Extract function name
        if ">" not in function_call_str:
            raise FunctionCallFormatError("Invalid function call format: Cannot find function name.")
        end_index = function_call_str.index(">")
        function_name = function_call_str[:end_index]
        param_config = self._get_arguments_config(function_name, tools)
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}
        for match_text in self.tool_call_parameter_regex.findall(parameters):
            if ">" not in match_text:
                raise FunctionCallFormatError(
                    f"Invalid function call format: Cannot find parameter name in tool call '{function_name}'."
                )
            idx = match_text.index(">")
            param_name = match_text[:idx]
            param_value = str(match_text[idx + 1 :])
            # Remove prefix and trailing \n
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            param_dict[param_name] = self._convert_param_value(param_value, param_name, param_config, function_name)

        function_call = OpenAIFunctionCallSchema(name=function_name, arguments=param_dict)
        tool_call = OpenAIFunctionToolCall(id=str(uuid.uuid4()), type="function", function=function_call)
        return tool_call

    def _get_function_calls(self, model_output: str) -> list[str]:
        """Return ``<function=...>`` bodies found inside ``<tool_call>`` blocks.
        Empty list = nothing recoverable (caller treats as "no tool calls").
        """
        matched_ranges = self.tool_call_regex.findall(model_output)
        raw_tool_calls = [match[0] if match[0] else match[1] for match in matched_ranges]
        raw_function_calls = []
        for tool_call in raw_tool_calls:
            raw_function_calls.extend(self.tool_call_function_regex.findall(tool_call))
        return [match[0] if match[0] else match[1] for match in raw_function_calls]

    def extract_tool_calls(
        self, model_output: str, tools: list[OpenAIFunctionToolSchema]
    ) -> tuple[str, list[OpenAIFunctionToolCall]]:
        """Parse ``<tool_call>...</tool_call>`` blocks out of ``model_output``.

        Returns ``(content_before_marker, tool_calls)``. ``tool_calls`` is
        ``[]`` whenever nothing parseable comes out -- whether the marker
        was absent or present-but-unrecoverable. Callers decide what to
        do (single-shot raises, chat treats as turn-end). When a function
        name IS recovered but invalid (unknown name, bad arg type, ...)
        we still raise :class:`FunctionCallFormatError`.
        """
        if self.tool_call_start_token not in model_output:
            return model_output, []

        function_calls = self._get_function_calls(model_output)
        if not function_calls:
            return model_output, []
        tool_calls = [self._parse_xml_function_call(function_call_str, tools) for function_call_str in function_calls]

        content_index = model_output.find(self.tool_call_start_token)
        content_index = content_index if content_index >= 0 else model_output.find(self.tool_call_prefix)
        content = model_output[:content_index]

        return content, tool_calls


class HermesToolParser:
    """Parser for the Hermes JSON tool-call format.
    Expected format::
        <tool_call>
        {"name": "<function-name>", "arguments": {...}}
        </tool_call>
    """

    _FORMAT_HINT = (
        'Expected format:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
    )

    def __init__(self):
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_regex = regex.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", regex.DOTALL)

    def extract_tool_calls(
        self, model_output: str, tools: list[OpenAIFunctionToolSchema]
    ) -> tuple[str, list[OpenAIFunctionToolCall]]:
        """Parse ``<tool_call>{...}</tool_call>`` JSON blocks out of ``model_output``.

        Returns ``(content_before_marker, tool_calls)``. ``tool_calls`` is
        ``[]`` when nothing parseable comes out (no marker, or marker
        pair with empty body); callers decide. The "unclosed" case
        (``<tool_call>`` opened with partial JSON, no closing tag) DOES
        raise -- partial JSON is a clear formatting bug worth surfacing
        back to the model.
        """
        if self.tool_call_start_token not in model_output:
            return model_output, []
        if self.tool_call_end_token not in model_output:
            raise FunctionCallFormatError(
                f"Unclosed tool call: missing {self.tool_call_end_token}. {self._FORMAT_HINT}"
            )

        matches = [m for m in self.tool_call_regex.findall(model_output) if m.strip()]
        if not matches:
            return model_output, []

        valid_names = {tool.function.name for tool in tools if tool.type == "function"}

        tool_calls: list[OpenAIFunctionToolCall] = []
        for raw in matches:
            tool_calls.append(self._parse_single(raw, valid_names))

        content_index = model_output.find(self.tool_call_start_token)
        return model_output[:content_index], tool_calls

    def _parse_single(self, raw: str, valid_names: set[str]) -> OpenAIFunctionToolCall:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise FunctionCallFormatError(
                f"Invalid tool_call JSON: {e.msg} (line {e.lineno} column {e.colno}). {self._FORMAT_HINT}"
            ) from None

        if not isinstance(obj, dict):
            raise FunctionCallFormatError(
                f"Invalid tool_call: expected a JSON object, got {type(obj).__name__}. {self._FORMAT_HINT}"
            )
        if "name" not in obj:
            raise FunctionCallFormatError(f"Invalid tool_call: missing 'name' field. {self._FORMAT_HINT}")

        name = obj["name"]
        if not isinstance(name, str):
            raise FunctionCallFormatError(f"Invalid tool_call: 'name' must be a string, got {type(name).__name__}.")
        if name not in valid_names:
            raise FunctionCallFormatError(
                f"Invalid action: function '{name}' is not defined in the tools list.\n"
                f"Allowed functions should be one of: {sorted(valid_names)}."
            )

        arguments: Any = obj.get("arguments", {})
        if arguments is None:
            arguments = {}
        if isinstance(arguments, str):
            # Some models double-encode arguments as a JSON string; accept that.
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                raise FunctionCallFormatError(
                    f"Invalid arguments JSON for '{name}': {e.msg} (line {e.lineno} column {e.colno})."
                ) from None
        if not isinstance(arguments, dict):
            raise FunctionCallFormatError(
                f"Invalid arguments for '{name}': expected a JSON object, got {type(arguments).__name__}."
            )

        function_call = OpenAIFunctionCallSchema(name=name, arguments=arguments)
        return OpenAIFunctionToolCall(id=str(uuid.uuid4()), type="function", function=function_call)


_PARSER_REGISTRY: dict[str, type] = {
    "qwen3_coder": XMLToolParser,
    "hermes": HermesToolParser,
}


def get_tool_parser(name: str):
    """Instantiate a tool-call parser by registered name."""
    if name not in _PARSER_REGISTRY:
        raise ValueError(f"Unknown tool parser: {name!r}. Available parsers: {sorted(_PARSER_REGISTRY.keys())}.")
    return _PARSER_REGISTRY[name]()
