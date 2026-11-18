"""Regression test ensuring ``uni_agent.gateway.session`` binds pure types before importing ``.codec``."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def test_session_binds_trajectory_before_importing_codec(tmp_path):
    """Simulates ``codec -> verl -> VERL_USE_EXTERNAL_MODULES -> uni_agent.gateway.session.Trajectory``."""
    # Inject a minimal ``verl`` package whose ``__init__`` re-entrantly imports ``Trajectory``
    # from ``uni_agent.gateway.session``, matching the VERL_USE_EXTERNAL_MODULES cycle.
    tok_pkg = tmp_path / "verl" / "utils" / "tokenizer"
    tok_pkg.mkdir(parents=True)
    (tmp_path / "verl" / "__init__.py").write_text(
        "from uni_agent.gateway.session import Trajectory\nREENTRANT_TRAJECTORY = Trajectory\n",
        encoding="utf-8",
    )
    (tmp_path / "verl" / "utils" / "__init__.py").write_text("", encoding="utf-8")
    (tok_pkg / "__init__.py").write_text("def normalize_token_ids(x):\n    return list(x)\n", encoding="utf-8")
    (tok_pkg / "chat_template.py").write_text(
        "def apply_chat_template(*a, **kw):\n    return []\n"
        "def initialize_turn_separator(*a, **kw):\n    return None\n",
        encoding="utf-8",
    )
    (tok_pkg / "continuous_token.py").write_text(
        "class MergeResult:\n    pass\nclass VLContinuousTokenMixin:\n    pass\n",
        encoding="utf-8",
    )
    (tok_pkg / "continuous_token_wiring.py").write_text(
        "def create_continuous_token_builder(*a, **kw):\n    return None\n",
        encoding="utf-8",
    )
    (tmp_path / "verl" / "utils" / "rollout_trace.py").write_text(
        "class RolloutTraceConfig:\n    @classmethod\n    def get_instance(cls):\n        return None\n",
        encoding="utf-8",
    )
    (tmp_path / "verl" / "utils" / "tracking.py").write_text(
        "class RLInsightLogger:\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "verl" / "utils" / "net_utils.py").write_text(
        "def is_valid_ipv6_address(addr):\n    return False\n",
        encoding="utf-8",
    )
    rollout_pkg = tmp_path / "verl" / "workers" / "rollout"
    rollout_pkg.mkdir(parents=True)
    (tmp_path / "verl" / "workers" / "__init__.py").write_text("", encoding="utf-8")
    (rollout_pkg / "__init__.py").write_text("", encoding="utf-8")
    (rollout_pkg / "utils.py").write_text(
        "def run_uvicorn(*a, **kw):\n    return None\n",
        encoding="utf-8",
    )
    (rollout_pkg / "llm_server.py").write_text(
        "class LLMServerClient:\n    pass\n",
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(repo_root)])

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            ("import uni_agent.gateway.session as s\nimport verl\nassert verl.REENTRANT_TRAJECTORY is s.Trajectory\n"),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Re-entrant import failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
