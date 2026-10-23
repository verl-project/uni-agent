"""Exercise a model checker with a real stdio child and local HTTP requests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("execute_tool", [True, False])
@pytest.mark.parametrize("stream", [True, False])
def test_model_check_requires_tool_execution_and_tool_reply(tmp_path, monkeypatch, capsys, execute_tool, stream):
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        """import json,subprocess,sys,urllib.request
config = json.load(sys.stdin)
url = config['model']['base_url'] + '/chat/completions'
def post(payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req) as response:
        text = response.read().decode()
    if not payload['stream']:
        return json.loads(text)['choices'][0]['message']
    event = json.loads(text.split('data: ',1)[1].split('\\n\\n',1)[0])
    return event['choices'][0]['delta']
payload = {'stream': STREAM, 'tools':[{'type':'function','function':{'name':'bash'}}], 'messages':config['messages']}
message = post(payload)
if EXECUTE_TOOL:
    call = message['tool_calls'][0]
    subprocess.run(json.loads(call['function']['arguments'])['command'], shell=True, check=True)
    payload['messages'] = [{'role':'tool','tool_call_id':call['id'],'content':'command completed'}]
    post(payload)
print(json.dumps({'output':{'content':'done'},'finished':True}))
""".replace("EXECUTE_TOOL", repr(execute_tool)).replace("STREAM", repr(stream))
    )
    case = tmp_path / "case.json"
    case.write_text(
        json.dumps(
            {
                "recipe": "model-tool",
                "entrypoint": [sys.executable, str(wrapper)],
                "probe_argv": [sys.executable, "--version"],
            }
        )
    )
    spec = importlib.util.spec_from_file_location("sidecar_model_check", ROOT / "examples/sidecar/check_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "Path", lambda value: tmp_path / "testbed" if value == "/testbed" else Path(value))
    monkeypatch.setattr(sys, "argv", ["check_model.py", "--case", str(case)])
    if execute_tool:
        module.main()
        report = json.loads(capsys.readouterr().out)
        assert report["result"] == "passed"
        assert report["model_calls"] == 2
    else:
        with pytest.raises(RuntimeError, match="not both observed"):
            module.main()
    assert not list((tmp_path / "testbed").glob(".sidecar-check-*"))
