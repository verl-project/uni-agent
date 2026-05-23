# ruff: noqa: E501
import os
import shlex
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uni_agent.interaction import AgentEnv, AgentEnvConfig
from uni_agent.tools import ToolConfig

# create environment
run_id = str(uuid.uuid4())
impl = os.getenv("DEPLOYMENT", "vefaas").lower()

if impl == "local":
    deployment_config = {
        "type": "local",
        "image": os.getenv("LOCAL_DEPLOYMENT_IMAGE", "python:3.12"),
        "command": os.getenv(
            "LOCAL_DEPLOYMENT_COMMAND",
            "python3 -m pip install -q swe-rex && "
            "python3 -m swerex.server --host 0.0.0.0 --port {port} --auth-token {token}",
        ),
        "timeout": 300.0,
        "startup_timeout": 180.0,
    }
    local_runtime = os.getenv("LOCAL_CONTAINER_RUNTIME")
    local_network = os.getenv("LOCAL_DEPLOYMENT_NETWORK")
    local_host = os.getenv("LOCAL_DEPLOYMENT_HOST")
    local_port = os.getenv("LOCAL_DEPLOYMENT_PORT")
    local_extra_args = os.getenv("LOCAL_DEPLOYMENT_EXTRA_ARGS")
    if local_runtime:
        deployment_config["container_runtime"] = local_runtime
    if local_network:
        deployment_config["network"] = local_network
    if local_host:
        deployment_config["host"] = local_host
    if local_port:
        deployment_config["published_port"] = int(local_port)
    if local_extra_args:
        deployment_config["extra_run_args"] = shlex.split(local_extra_args)
elif impl == "local_attach":
    # Attach to a user-managed sandbox.
    # start the sandbox:
    #   docker run -d --name milo-sandbox -p 18000:18000 -v ~/.uni-agent/app/milo:/workspace
    #     python:3.12 bash -lc "pip install -q swe-rex && \
    #     python3 -m swerex.server --host 0.0.0.0 --port 18000 --auth-token milowww"
    deployment_config = {
        "type": "local_attach",
        "host": os.getenv("LOCAL_ATTACH_HOST", "http://127.0.0.1"),
        "port": int(os.getenv("LOCAL_ATTACH_PORT", "18000")),
        "auth_token": os.environ.get("LOCAL_ATTACH_AUTH_TOKEN", "milowww"),
        "timeout": 60.0,
        "startup_timeout": 30.0,
    }
elif impl == "vefaas":
    assert os.getenv("VOLCE_ACCESS_KEY") is not None, "VOLCE_ACCESS_KEY must be set"
    assert os.getenv("VOLCE_SECRET_KEY") is not None, "VOLCE_SECRET_KEY must be set"
    deployment_config = {
        "type": "vefaas",
        "image": "enterprise-public-2-cn-beijing.cr.volces.com/vefaas-public/python:3.12",
        "command": "curl -fsSL https://vefaas-swe.tos-cn-beijing.ivolces.com/swe-rex/install_1.4.0.sh | bash -s -- {token}",
        "timeout": 300.0,
        "startup_timeout": 180.0,
        # "proxy": "xxxxxx",
    }
elif impl == "modal":
    deployment_config = {
        "type": "modal",
        "image": "python:3.12",
        "startup_timeout": 600.0,
        "runtime_timeout": 300.0,
        "deployment_timeout": 3600.0,
    }
elif impl == "":
    raise ValueError("DEPLOYMENT must be set")
else:
    raise ValueError(f"Invalid environment implementation: {impl}")

env_config = {
    "deployment": deployment_config,
    "env_variables": {
        "PIP_PROGRESS_BAR": "off",
    },
}
env_config = AgentEnvConfig(**env_config)
env = AgentEnv(run_id=run_id, env_config=env_config)
env.start()

# install tools in the environment
tools_config = [
    {"name": "execute_bash"},
    {"name": "str_replace_editor"},
]
tools = [ToolConfig(**tool_config).get_tool() for tool_config in tools_config]
env.install_tools(tools)
out = env.communicate("which str_replace_editor")
print(f"[Tool check] which str_replace_editor\n  -> {out.strip()}\n")

# --- Simple sandbox demo: create script -> run -> output to file -> cat (shows persistence) ---
print("=" * 60)
print("  Sandbox demo: create script -> run -> output to file -> cat")
print("=" * 60)

# 1. Install dependency (persists in this sandbox)
print("\n[Step 1] Install numpy")
env.communicate("pip install numpy -q")
print("  -> done\n")

# 2. Create a runnable script with str_replace_editor (writes result to /tmp/demo_out.txt)
_script = "import numpy as np; print(np.array([1,2,3]).sum())"
print("[Step 2] str_replace_editor create /tmp/demo.py")
env.communicate(f"str_replace_editor create --path /tmp/demo.py --file_text {shlex.quote(_script)}")
print("  -> done\n")

# 3. Run the script (output goes to /tmp/demo_out.txt)
print("[Step 3] Run script (python3 /tmp/demo.py > /tmp/demo_out.txt)")
env.communicate("execute_bash 'python3 /tmp/demo.py > /tmp/demo_out.txt'")
print("  -> done\n")

# 4. Cat the output path
print("[Step 4] cat /tmp/demo_out.txt")
out = env.communicate("cat /tmp/demo_out.txt")
print(f"  -> {out.strip()}\n")

print("=" * 60)
print("  Demo done (sandbox: script + output file persisted)")
print("=" * 60)

env.close()
