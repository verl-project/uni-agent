"""Simulated sandbox deployment for performance testing.

The LLM (vLLM replicas + router) runs for real; only the sandbox
(docker/swe-rex bash execution) is stubbed. ``SimulatedRuntime`` implements
swerex's ``AbstractRuntime`` so the entire production code path
(``AgentEnv`` -> tools install -> ``run_action``) runs unmodified -- only
the leaf bash execution returns a representative canned observation.

Built test-first. Two responsibilities so far: the command router
(``_route``) and observation rendering with reproducible seeding.
"""

from __future__ import annotations

import hashlib
import os
import random
import re

from swerex.runtime.abstract import (
    AbstractRuntime,
    BashAction,
    BashInterruptAction,
    Observation,
)

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
# Patterns checked in order; first match wins. The command string is the bash
# that ``tools_manager.get_tool_bash_command`` emits (see findings.md).
#
# Install-phase commands (which/export/chmod/mkdir/pip) must route to
# ``install`` so they no-op successfully and ``AgentEnv.install_tools`` passes.
_ROUTE_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("finish", re.compile(r"""echo\s+['"]<<<Finished>>>['"]""")),
    ("install", re.compile(r"^(which|export|chmod|mkdir|pip|pip3)\b")),
    ("install", re.compile(r"\bpip3?\s+install\b")),
    ("install", re.compile(r"\bpython\d?\s+-m\s+pip\b")),
    ("editor:view", re.compile(r"^str_replace_editor\b.*--command\s+view\b")),
    ("editor:create", re.compile(r"^str_replace_editor\b.*--command\s+create\b")),
    ("editor:str_replace", re.compile(r"^str_replace_editor\b.*--command\s+str_replace\b")),
    ("editor:insert", re.compile(r"^str_replace_editor\b.*--command\s+insert\b")),
    ("editor:undo_edit", re.compile(r"^str_replace_editor\b.*--command\s+undo_edit\b")),
    ("test_output", re.compile(r"^(\S*python\S*\s+-m\s+pytest\b|^pytest\b)")),
    ("python_script", re.compile(r"^python\d?\s")),
    ("listing", re.compile(r"^(find|ls)\b")),
    ("search", re.compile(r"^grep\b")),
    ("file_view", re.compile(r"^(cat|head|tail)\b")),
]


# ---------------------------------------------------------------------------
# Template pool
# ---------------------------------------------------------------------------
# Each route key -> list of (weight, text). Weights are hand-tuned to a
# realistic success:failure ratio (~8:2) and length spread -- NOT copied from
# the run logs, whose frequency is polluted by an unrelated KV-memory bug
# (see findings.md). Text structure follows real SWE-bench samples.
#
# finish / install are fixed single outputs (not sampled).

_FINISH_OUTPUT = "<<<Finished>>>"

_TEMPLATES: dict[str, list[tuple[int, str]]] = {
    "editor:view": [
        (3, "Here's the result of running `cat -n` on /testbed/x.py:\n"
            "     1\timport os\n"
            "     2\t\n"
            "     3\tclass Filter:\n"
            "     4\t    def __init__(self, model):\n"
            "     5\t        self.model = model\n"),
        (2, "Here's the result of running `cat -n` on /testbed/models.py:\n"
            "     1\tclass Model:\n"
            "     2\t    objects = Manager()\n"),
    ],
    "editor:create": [
        (5, "File created successfully at: /testbed/reproduce_issue.py"),
        (1, "ERROR: file already exists at /testbed/x.py. Use str_replace to edit it."),
    ],
    "editor:str_replace": [
        (4, "The file /testbed/x.py has been edited. Here's the result of running `cat -n` on a snippet:\n"
            "    10\t    def filter(self, qs):\n"
            "    11\t        return qs\n"),
        (1, "ERROR: old_str was not found in /testbed/x.py. Make sure it matches exactly."),
    ],
    "editor:insert": [
        (5, "The file /testbed/x.py has been edited. Inserted text after line 11."),
    ],
    "editor:undo_edit": [
        (5, "Last edit to /testbed/x.py has been reverted."),
    ],
    "test_output": [
        (4, "============================= test session starts ==============================\n"
            "platform linux -- Python 3.9.20, pytest-7.4.0\n"
            "rootdir: /testbed\n"
            "collected 5 items\n\n"
            "tests/test_x.py .....                                                 [100%]\n\n"
            "============================== 5 passed in 2.13s ===============================\n"),
        (3, "============================= test session starts ==============================\n"
            "collected 5 items\n\n"
            "tests/test_x.py ..F..                                                [100%]\n\n"
            "=================================== FAILURES ===================================\n"
            "_____________________________ test_third ______________________________________\n"
            "    assert result == expected\n"
            "E   AssertionError: assert 3 == 5\n"
            "=========================== short test summary info ===========================\n"
            "FAILED tests/test_x.py::test_third - AssertionError: assert 3 == 5\n"
            "========================= 1 failed, 4 passed in 2.10s ==========================\n"),
    ],
    "python_script": [
        (5, "ok\n"),
        (2, "Traceback (most recent call last):\n"
            "  File \"/testbed/reproduce_issue.py\", line 5, in <module>\n"
            "    obj = load(path)\n"
            "  File \"/testbed/pkg/loader.py\", line 21, in load\n"
            "    return _read(path)\n"
            "FileNotFoundError: [Errno 2] No such file or directory: '/testbed/data.bin'\n"),
        (1, "/bin/bash: line 1: python: command not found\n"),
    ],
    "listing": [
        (3, "/testbed/setup.py\n/testbed/pkg/__init__.py\n/testbed/pkg/models.py\n/testbed/pkg/views.py\n"),
        (1, "/testbed\n"),
    ],
    "search": [
        (3, "/testbed/x.py:12:    def filter(self, qs):\n/testbed/y.py:40:    qs = filter(qs)\n"),
        (2, "Your command ran successfully and did not produce any output.\n"),
    ],
    "file_view": [
        (3, "import os\n\n\nclass Model:\n    pass\n"),
        (1, "cat: /testbed/x.py: No such file or directory\n"),
    ],
    "default": [
        (4, "Your command ran successfully and did not produce any output.\n"),
        (1, "done\n"),
    ],
}


# Bundled YAML (real-sample structure, hand-tuned weights). Resolved relative
# to this file so it works regardless of CWD.
_DEFAULT_TEMPLATES_PATH = os.path.join(os.path.dirname(__file__), "observations.yaml")


def load_templates(path: str | None = None) -> dict[str, list[tuple[int, str]]]:
    """Load the observation template pool from YAML.

    Schema: ``route_key -> [{weight: int, text: str}, ...]``. ``path=None`` loads
    the bundled default. Falls back to the in-source ``_TEMPLATES`` only if the
    YAML is unreadable, so a missing/corrupt file never crashes a perf run.
    """
    import yaml

    target = path or _DEFAULT_TEMPLATES_PATH
    try:
        raw = yaml.safe_load(_read_text(target))
    except (OSError, yaml.YAMLError):
        return {k: list(v) for k, v in _TEMPLATES.items()}
    pool: dict[str, list[tuple[int, str]]] = {}
    for key, entries in (raw or {}).items():
        out: list[tuple[int, str]] = []
        for entry in entries:
            out.append((int(entry["weight"]), str(entry["text"])))
        pool[key] = out
    return pool


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _scale_text(text: str, scale: float) -> str:
    """Stretch (or shrink) ``text`` toward ``len(text) * scale`` chars.

    Grown by repeating a clean newline-terminated block; shrunk by truncating
    on a line boundary. A no-op at scale == 1.0.
    """
    if scale <= 0:
        return ""
    target = int(len(text) * scale)
    if target <= len(text):
        return text[:target].rsplit("\n", 1)[0]
    if not text:
        return text
    block = text if text.endswith("\n") else text + "\n"
    reps = max(1, target // max(1, len(block)) + 1)
    grown = (block * reps)[:target]
    if not grown.endswith("\n"):
        grown = grown.rsplit("\n", 1)[0]
    return grown


class SimulatedRuntime(AbstractRuntime):
    """CPU-only runtime that returns canned observations by routing the
    command string. Subclassed from ``AbstractRuntime`` for protocol
    compatibility; no real session is created."""

    def __init__(
        self,
        run_id: str = "simulated",
        *,
        seed: int | None = None,
        observation_scale: float = 1.0,
        templates: dict[str, list[tuple[int, str]]] | None = None,
        templates_path: str | None = None,
        timeout: "TimeoutSimConfig | None" = None,
        terminal_dead: "TerminalDeadConfig | None" = None,
    ) -> None:
        from .config import TerminalDeadConfig, TimeoutSimConfig

        self.run_id = run_id
        self._seed = seed
        self.observation_scale = observation_scale
        # Precedence: explicit dict > explicit yaml path > bundled default yaml.
        if templates is not None:
            self._templates = templates
        elif templates_path is not None:
            self._templates = load_templates(templates_path)
        else:
            self._templates = load_templates()
        self._timeout_cfg = timeout or TimeoutSimConfig()
        self._terminal_dead_cfg = terminal_dead or TerminalDeadConfig()
        # Failure state: once the terminal "dies" it stays dead -- every later
        # run_in_session (including env.py's liveness probe) returns no marker,
        # which is what makes env.py raise TerminalNotAliveError.
        self._dead = False
        # One independent RNG stream per route key, derived from
        # (global_seed, route_key). Same seed -> same per-key streams ->
        # reproducible sampling, without one command type's draws perturbing
        # another's. Failure modes get their own dedicated streams so timeout
        # and terminal_dead patterns are mutually reproducible/independent.
        self._rngs: dict[str, random.Random] = {}
        for key in self._templates:
            self._rngs[key] = random.Random(self._derive_seed(key))
        self._timeout_rng = random.Random(self._derive_seed("__timeout__"))
        self._dead_rng = random.Random(self._derive_seed("__terminal_dead__"))

    def _derive_seed(self, key: str) -> int | None:
        if self._seed is None:
            return None
        digest = hashlib.sha256(f"{self._seed}:{key}".encode()).digest()
        return int.from_bytes(digest[:8], "big")

    def _route(self, command: str) -> str:
        """Map a bash command string to a route key (a template group).

        Pure function of ``command``: same command always yields the same
        key. Order matters -- ``python -m pip install`` must hit ``install``
        before ``python_script``; ``str_replace_editor --command`` variants
        are split by subcommand.
        """
        stripped = command.strip()
        for key, pattern in _ROUTE_RULES:
            if pattern.search(stripped):
                return key
        return "default"

    def _render(self, route_key: str) -> str:
        """Sample a representative observation for ``route_key``.

        Fixed keys (finish/install) return constants; others sample by weight
        from the template pool, then apply ``observation_scale``.
        """
        if route_key == "finish":
            return _FINISH_OUTPUT
        if route_key == "install":
            return ""
        pool = self._templates.get(route_key)
        if not pool:
            # Unknown route key (e.g. an editor subcommand we didn't model):
            # fall back to the default pool rather than crash.
            pool = self._templates.get("default", [(1, "")])
        weights = [w for w, _ in pool]
        texts = [t for _, t in pool]
        rng = self._rngs.get(route_key)
        if rng is None:
            # A route key with no prebuilt stream (templates overridden to add
            # a new key): build an ephemeral one so sampling still works.
            rng = random.Random(self._derive_seed(route_key))
        text = rng.choices(texts, weights=weights, k=1)[0]
        return _scale_text(text, self.observation_scale)

    async def run_in_session(self, action) -> Observation:
        import asyncio

        from swerex.exceptions import CommandTimeoutError

        if isinstance(action, BashInterruptAction):
            # A dead terminal cannot accept an interrupt either -- this is what
            # pushes env.py out of its interrupt_session() happy path and into
            # the liveness-probe branch that raises TerminalNotAliveError.
            if self._dead:
                from swerex.exceptions import CommandTimeoutError

                raise CommandTimeoutError("simulated: terminal dead, interrupt unresponsive")
            return Observation(output="", exit_code=130)
        if not isinstance(action, BashAction):
            raise TypeError(f"Unsupported action type: {type(action)}")
        route_key = self._route(action.command)

        # Once the terminal is dead, every call (including env.py's liveness
        # probe ``echo 'terminal still alive'``) returns no marker -> env.py
        # concludes the terminal is dead and raises TerminalNotAliveError.
        if self._dead:
            return Observation(output="", exit_code=1)

        # Failure simulation only applies to real tool commands, not the
        # install/finish phase (which must stay reliable so install_tools
        # passes and submit terminates cleanly).
        if route_key not in ("install", "finish"):
            # terminal_dead: model it as a hung command. Time out (which makes
            # env.py probe liveness), then stay dead so the probe fails.
            if self._terminal_dead_cfg.enabled and self._dead_rng.random() < self._terminal_dead_cfg.probability:
                self._dead = True
                raise CommandTimeoutError("simulated: terminal died (command hung)")
            # timeout: recoverable. Sleep to mimic real bash wall-clock, then
            # raise so env.py decrements timeout_budget.
            if self._timeout_cfg.enabled and self._timeout_rng.random() < self._timeout_cfg.probability:
                if self._timeout_cfg.delay_seconds > 0:
                    await asyncio.sleep(self._timeout_cfg.delay_seconds)
                raise CommandTimeoutError("simulated: simulated command timeout")

        output = self._render(route_key)
        return Observation(output=output, exit_code=0)

    # --- AbstractRuntime protocol stubs (filled by later TDD cycles) ---------
    async def create_session(self, request):  # pragma: no cover - stub
        return None

    async def execute(self, command):  # pragma: no cover - stub
        from swerex.runtime.abstract import CommandResponse

        return CommandResponse(stdout="", stderr="", exit_code=0)

    async def upload(self, request):  # pragma: no cover - stub
        from swerex.runtime.abstract import UploadResponse

        return UploadResponse()

    async def read_file(self, request):  # pragma: no cover - stub
        from swerex.runtime.abstract import ReadFileResponse

        return ReadFileResponse(content="")

    async def write_file(self, request):  # pragma: no cover - stub
        from swerex.runtime.abstract import WriteFileResponse

        return WriteFileResponse()

    async def is_alive(self, *, timeout=None):  # pragma: no cover - stub
        from swerex.runtime.abstract import IsAliveResponse

        return IsAliveResponse(is_alive=True)

    async def close_session(self, request):  # pragma: no cover - stub
        from swerex.runtime.abstract import CloseSessionResponse

        return CloseSessionResponse()

    async def close(self):  # pragma: no cover - stub
        from swerex.runtime.abstract import CloseResponse

        return CloseResponse()


class SimulatedDeployment:
    """Deployment wrapper around :class:`SimulatedRuntime`.

    Implements swerex's ``AbstractDeployment`` protocol (structurally) so
    ``AgentEnv`` drives it exactly like docker/modal/etc. ``start()`` does
    nothing real -- no process, no docker, no swe-rex -- it just marks the
    runtime ready. The configured seed/scale flow into the SimulatedRuntime so
    sampling is reproducible before any command runs.
    """

    def __init__(
        self,
        run_id: str,
        *,
        seed: int | None = None,
        observation_scale: float = 1.0,
        timeout=None,
        terminal_dead=None,
    ) -> None:
        self.run_id = run_id
        self._runtime = SimulatedRuntime(
            run_id=run_id,
            seed=seed,
            observation_scale=observation_scale,
            timeout=timeout,
            terminal_dead=terminal_dead,
        )
        self._started = False

    @classmethod
    def from_config(cls, config, run_id: str | None = None) -> "SimulatedDeployment":
        return cls(
            run_id=run_id or "simulated",
            seed=config.seed,
            observation_scale=config.observation_scale,
            timeout=config.timeout,
            terminal_dead=config.terminal_dead,
        )

    def add_hook(self, hook):  # pragma: no cover - protocol no-op
        pass

    async def start(self, max_retries: int = 5) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def is_alive(self, *, timeout=None):
        from swerex.runtime.abstract import IsAliveResponse

        return IsAliveResponse(is_alive=self._started)

    @property
    def runtime(self) -> SimulatedRuntime:
        if not self._started:
            from swerex.exceptions import DeploymentNotStartedError

            raise DeploymentNotStartedError()
        return self._runtime

