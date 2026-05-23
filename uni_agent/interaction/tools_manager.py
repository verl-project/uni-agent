import json
import shlex

from pydantic import BaseModel, ConfigDict

from uni_agent.interaction.tool_parser import FunctionCallFormatError, get_tool_parser
from uni_agent.interaction.tool_schemas import (
    OpenAIFunctionCallSchema,
    OpenAIFunctionToolCall,
    OpenAIFunctionToolSchema,
)
from uni_agent.tools import ToolConfig


class ToolsManagerConfig(BaseModel):
    """Config for the tools list."""

    tools: list[ToolConfig]
    parser: str = "qwen3_coder"
    """Name of the registered tool-call parser. Built-in: "qwen3_coder", "hermes"."""
    model_config = ConfigDict(extra="ignore")


class ToolsManager:
    """Builds tool instances and OpenAI tool schemas from ToolsConfig."""

    def __init__(self, tools_manager_config: ToolsManagerConfig):
        self.tools_manager_config = tools_manager_config
        self.tools = [tc.get_tool() for tc in tools_manager_config.tools]
        self.tools_schemas = [t.get_tool_schema() for t in self.tools]
        self._tool_parser = get_tool_parser(tools_manager_config.parser)

    async def parse_action(
        self,
        model_output: str,
    ) -> tuple[str, list[OpenAIFunctionToolCall]]:
        """Parse tool calls from raw text. Returns ``(content, tool_calls)``;
        ``tool_calls`` is ``[]`` when the text contains no tool-call
        marker (callers decide -- single-shot raises, chat_mode treats
        as turn-end). Markers that ARE present but malformed raise
        :class:`FunctionCallFormatError`.
        """
        tools = [OpenAIFunctionToolSchema(**schema) for schema in self.tools_schemas]
        content, tool_calls = self._tool_parser.extract_tool_calls(model_output, tools)
        return content, tool_calls

    async def parse_structured_action(
        self,
        content: str,
        tool_calls_data: list[dict],
    ) -> tuple[str, list[OpenAIFunctionToolCall]]:
        """Parse OpenAI-style structured tool calls. May return an empty list
        (callers decide); unknown names / invalid JSON args raise
        :class:`FunctionCallFormatError`.
        """
        tool_calls = []
        valid_names = {schema["function"]["name"] for schema in self.tools_schemas}
        for tool_call_data in tool_calls_data:
            function_data = tool_call_data["function"]
            function_name = function_data["name"]
            if function_name not in valid_names:
                raise FunctionCallFormatError(
                    f"Invalid action: function '{function_name}' is not defined in the tools list.\n"
                    f"Allowed functions should be one of: {sorted(valid_names)}."
                )
            arguments = function_data.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise FunctionCallFormatError(
                        f"Invalid action: arguments for function '{function_name}' are not valid JSON."
                    ) from exc

            function_call = OpenAIFunctionCallSchema(name=function_name, arguments=arguments)
            tool_calls.append(
                OpenAIFunctionToolCall(
                    id=tool_call_data["id"],
                    type=tool_call_data.get("type", "function"),
                    function=function_call,
                )
            )
        return content, tool_calls

    def get_tool_bash_command(self, tool_call: OpenAIFunctionToolCall) -> str:
        function: OpenAIFunctionCallSchema = tool_call.function
        func_name: str = function.name
        func_params: dict = function.arguments

        if func_name == "submit":
            return "echo '<<<Finished>>>'"

        if func_name == "execute_bash":
            return func_params.get("command", "")

        if func_name == "lark-cli":
            command = func_params.get("command", "")
            return f"lark-cli {command}" if command else ""

        # Start building the command
        cmd_parts = [shlex.quote(func_name)]

        # If there's a 'command' parameter, put that next
        base_command = func_params.get("command")
        if base_command is not None:
            cmd_parts.append(shlex.quote(base_command))

        # Append all other parameters
        for param_key, param_value in func_params.items():
            if param_key == "command":
                continue

            # Use JSON for structured types so the script can json.loads them
            if isinstance(param_value, list | dict):
                param_str = json.dumps(param_value, ensure_ascii=False)
            else:
                param_str = str(param_value)
            cmd_parts.append(f"--{param_key}")
            cmd_parts.append(shlex.quote(param_str))

        return " ".join(cmd_parts)
