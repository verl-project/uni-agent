#!/usr/bin/env bash
# Run from the checkout root, with Python deps and OPENCODE_BIN provisioned.
set -euo pipefail
out=${1:?usage: run_opencode_integration.sh OUTPUT_DIRECTORY}
mkdir -p "$out"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OPENCODE_VERSION=${OPENCODE_VERSION:-1.19.3}
: "${OPENCODE_BIN:?Set the provisioned OpenCode executable}"
{
  git rev-parse HEAD
  git status --porcelain
  uname -a
  python --version
  "$OPENCODE_BIN" --version
  python -c 'from importlib.metadata import distributions; print("\n".join(sorted(d.metadata["Name"] + "==" + d.version for d in distributions())))'
} > "$out/environment.txt"
python -m pytest \
  tests/uni_agent/agents/test_opencode_agent.py \
  tests/uni_agent/agents/test_model_config.py \
  tests/uni_agent/agents/test_openclaw_agent.py \
  tests/uni_agent/agents/test_claude_code_agent.py \
  tests/uni_agent/agents/test_mini_swe_agent_agent.py \
  tests/uni_agent/agents/test_opencode_cli_probe.py \
  tests/uni_agent/agents/test_opencode_process_cleanup.py \
  tests/uni_agent/agents/test_opencode_task_smoke.py \
  tests/uni_agent/sandbox/test_local_sandbox.py \
  tests/uni_agent/sandbox/test_local_descendant_groups.py \
  -q -s --tb=short --junitxml="$out/tests.xml" 2>&1 | tee "$out/tests.log"
python - "$out" <<'PY'
import json
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

out = pathlib.Path(sys.argv[1])
suites = ET.parse(out / 'tests.xml').getroot().iter('testsuite')
counts = {key: 0 for key in ('tests', 'failures', 'errors', 'skipped')}
for suite in suites:
    for key in counts:
        counts[key] += int(suite.attrib.get(key, 0))
assert counts['tests'] == 193, counts
assert not any(counts[k] for k in ('failures', 'errors', 'skipped')), counts
records = [json.loads(m) for m in re.findall(r'opencode_process_cleanup=(\{[^\n]+\})', (out / 'tests.log').read_text())]
assert len(records) == 4, records
assert all(not r['live_before_rescue'] and not r['rescue_signals'] and not r['cleanup_errors'] for r in records)
(out / 'summary.json').write_text(json.dumps({'counts': counts, 'cli_cleanup': records}, indent=2) + '\n')
PY
