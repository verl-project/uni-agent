# ruff: noqa: E501
import os
import uuid

from uni_agent.interaction import (
    AgentEnv,
    AgentEnvConfig,
    AgentInteraction,
    OpenAICompatibleChatModel,
    ToolsManager,
    ToolsManagerConfig,
)
from uni_agent.tools import ToolConfig

run_id = str(uuid.uuid4())
user_request = (
    "Please search arXiv papers from the last month about 'Agent Reinforcement Learning', read the abstracts, "
    "and give me a ranked list of the 5 most relevant papers with one-sentence reasons. "
    "For each paper, include the arXiv abstract URL."
)

impl = os.getenv("DEPLOYMENT", "vefaas").lower()
if impl == "local":
    raise NotImplementedError("Local deployment is not implemented yet")
if impl != "vefaas":
    raise ValueError(f"Invalid environment implementation: {impl}")

access_key = os.getenv("VOLCE_ACCESS_KEY")
secret_key = os.getenv("VOLCE_SECRET_KEY")
function_id = os.getenv("VEFAAS_FUNCTION_ID")
function_route = os.getenv("VEFAAS_FUNCTION_ROUTE")
model_base_url = os.getenv("BASE_URL", "http://localhost:8000/v1")
model_api_key = os.getenv("API_KEY", "EMPTY")
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-Coder-30B-A3B-Instruct")

print("=" * 80)
print("Customize a simple search agent")
print("=" * 80)
print(f"Run ID: {run_id}")
print(f"User request: {user_request}")
print(f"Model endpoint: {model_base_url}")
print(f"Model name: {model_name}")
print(f"API key configured: {model_api_key != 'EMPTY'}")

env_config = {
    "deployment": {
        "type": "vefaas",
        "image": "enterprise-public-2-cn-beijing.cr.volces.com/vefaas-public/python:3.12",
        "command": "curl -fsSL https://vefaas-swe.tos-cn-beijing.ivolces.com/swe-rex/install_1.4.0.sh | bash -s -- {token}",
        "timeout": 300.0,
        "startup_timeout": 180.0,
        "function_id": function_id,
        "function_route": function_route,
    },
    "env_variables": {
        "PIP_PROGRESS_BAR": "off",
    },
}
env = AgentEnv(run_id=run_id, env_config=AgentEnvConfig(**env_config))

tools_manager = ToolsManager(
    ToolsManagerConfig(
        tools=[
            ToolConfig(name="search_arxiv"),
            ToolConfig(name="finish"),
        ]
    )
)
model = OpenAICompatibleChatModel(
    base_url=model_base_url,
    api_key=model_api_key,
    model_name=model_name,
    sampling_params={"temperature": 0.0, "max_tokens": 8192},
)
model.set_tools_schemas(tools_manager.tools_schemas)

messages = [
    {
        "role": "system",
        "content": (
            "You are a simple arXiv paper search agent. "
            "Every assistant response MUST contain EXACTLY ONE tool call. "
            "Do not reply with plain text without a tool call."
        ),
    },
    {"role": "user", "content": user_request},
]

interaction = AgentInteraction(
    run_id=run_id,
    env=env,
    model=model,
    tools_manager=tools_manager,
    messages=messages,
    action_timeout=60,
    max_turns=20,
)


print("\n[1/4] Starting environment...")
env.start()

print("[2/4] Installing tools...")
env.install_tools(tools_manager.tools)
tool_check = env.communicate("which search_arxiv && which finish")
print(tool_check.strip())

print("\n[3/4] Running interaction...")
result = interaction.run()
last_step = result["trajectory"][-1] if result["trajectory"] else None

print("\n[4/4] Final status:")
if last_step is not None:
    print(f"exit_reason: {last_step.exit_reason}")
    print(f"done: {last_step.done}")
else:
    print("No step output found.")

print("\nFinal result:")
if last_step is None:
    print("(empty)")
elif last_step.tool_results:
    print(last_step.tool_results[-1].observation)
else:
    print(last_step.response or "(empty)")

env.close()
