"""Recipe-owned coding smoke dataset in the existing TerminalBenchTask schema.

This is a synthetic acceptance fixture, not a public benchmark result. Tests are
only injected by the task verifier after the agent exits.
"""
import argparse
import base64
import io
import json
from pathlib import Path
import tarfile

PROMPT = """Create /workspace/solution.py with a function merge_intervals(intervals).
Input is a list of integer pairs [start, end] with start <= end. Return sorted,
non-overlapping intervals as lists, merging overlaps and touching endpoints.
Do not mutate the input. Empty input returns []. Use only the Python standard
library. Use your exec tool to write the module and test it, then report completion.
"""

TESTS = '''#!/bin/sh
set -eu
python3 - <<'PY'
import importlib.util, json
spec = importlib.util.spec_from_file_location("candidate", "/workspace/solution.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cases = [([], []), ([[1,3],[2,6],[8,10],[15,18]], [[1,6],[8,10],[15,18]]),
         ([[5,7],[1,5]], [[1,7]]), ([[1,4],[2,3]], [[1,4]]),
         ([[-4,-1],[-1,0],[2,2]], [[-4,0],[2,2]]), ([[3,3],[3,3]], [[3,3]])]
for value, expected in cases:
    before = json.dumps(value)
    assert module.merge_intervals(value) == expected
    assert json.dumps(value) == before, "input was mutated"
# Deterministic many-case reference check, independent of the usual merge algorithm.
import random
rng = random.Random(941)
for _ in range(100):
    pairs = [sorted([rng.randrange(-10,11), rng.randrange(-10,11)]) for _ in range(rng.randrange(10))]
    expected = []
    for pair in sorted(pairs):
        if expected and pair[0] <= expected[-1][1]:
            expected[-1][1] = max(expected[-1][1], pair[1])
        else:
            expected.append(pair[:])
    before = json.dumps(pairs)
    assert module.merge_intervals(pairs) == expected
    assert json.dumps(pairs) == before
with open("/logs/verifier/reward.txt", "w") as f:
    f.write("1")
print("106 cases passed")
PY
'''


CASES = {
    "merge_intervals": (PROMPT, TESTS),
    "unicode_counts": (
        "请创建 /workspace/solution.py，实现 count_words(text)。用正则表达式按 Unicode 字母数字组成的词分词，"
        "下划线不属于词；对每个词先做 Unicode NFC 规范化，再 casefold，统计出现次数，返回字典。"
        "请使用 re.findall(r'[^\\W_]+', text) 的分词语义，不要改为按空格拆分。空输入返回空字典。"
        "只用 Python 标准库，用 exec 写文件并测试。",
        r"""#!/bin/sh
set -eu
python3 - <<'PY'
import importlib.util, re, unicodedata
spec=importlib.util.spec_from_file_location('candidate','/workspace/solution.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
values=['', 'Hello HELLO 世界 世界', 'a_b a-b', 'Straße STRASSE', 'CAFÉ café', '123 123 x2',
        '中文，测试！中文', 'a\tB\nA', 'e\u0301 é', 'ＡＢＣ abc', '🙂 hello🙂HELLO']
for text in values:
    expected={}
    for token in re.findall(r'[^\W_]+',text):
        key=unicodedata.normalize('NFC',token).casefold()
        expected[key]=expected.get(key,0)+1
    assert m.count_words(text)==expected, repr(text)
open('/logs/verifier/reward.txt','w').write('1')
print('11 Unicode cases passed')
PY
"""),
    "topological_sort": (
        "Create /workspace/solution.py implementing topological_sort(graph). graph is a dict mapping string nodes "
        "to lists of outgoing neighbors. Include nodes appearing only as neighbors. Return the lexicographically "
        "smallest valid topological ordering as a list. Duplicate edges count once. Raise ValueError on any cycle "
        "including self-loops. Do not mutate the input. Empty graph returns []. Use standard library only and exec to test.",
        r"""#!/bin/sh
set -eu
python3 - <<'PY'
import importlib.util, itertools, json, random
spec=importlib.util.spec_from_file_location('candidate','/workspace/solution.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
cases=[{}, {'a':['b','b']}, {'z':['a'],'b':[]}, {'a':['a']}, {'a':['b'],'b':['a']}]
rng=random.Random(405)
for _ in range(35):
    nodes=list('abcde')[:rng.randrange(1,6)]
    cases.append({n:[v for v in nodes if rng.random()<0.2] for n in nodes})
for graph in cases:
    before=json.dumps(graph)
    nodes=sorted(set(graph)|{v for values in graph.values() for v in values})
    expected=None
    for perm in itertools.permutations(nodes):
        pos={n:i for i,n in enumerate(perm)}
        if all(pos[a]<pos[b] for a,bs in graph.items() for b in bs):
            expected=list(perm);break
    if expected is None:
        try: m.topological_sort(graph)
        except ValueError: pass
        else: raise AssertionError('cycle must raise')
    else: assert m.topological_sort(graph)==expected, repr(graph)
    assert json.dumps(graph)==before
open('/logs/verifier/reward.txt','w').write('1')
print('40 graph cases passed')
PY
"""),
}


def task_payload(image="openclaw-recipe-runtime:2026.9.2", case="merge_intervals"):
    prompt, tests = CASES[case]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        data = tests.encode()
        entry = tarfile.TarInfo("test.sh")
        entry.size, entry.mode, entry.mtime = len(data), 0o755, 0
        archive.addfile(entry, io.BytesIO(data))
    return {
        "name": "terminal_bench",
        "sandbox": {"provider": "docker", "image": image,
                    "sandbox_kwargs": {"pull_policy": "never", "run_args": ["--network", "host"]}},
        "agent": {"name": "openclaw", "cli_timeout_seconds": 240, "run_timeout": 270},
        "prompt": [{"role": "user", "content": prompt}],
        "metadata": {"instance_id": "openclaw-" + case.replace("_", "-") + "-v1", "dataset_version": "recipe-smoke-v1",
                     "agent_timeout": 330, "verifier_timeout": 30,
                     "environment_json": json.dumps({"workdir": "/workspace", "allow_internet": True}),
                     "verifier_env_json": "{}", "tests_archive": base64.b64encode(buffer.getvalue()).decode()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--format", choices=["json", "parquet"], default="json")
    parser.add_argument("--case", choices=sorted(CASES), default="merge_intervals")
    args = parser.parse_args()
    task = task_payload(case=args.case)
    if args.format == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        row = {"data_source": "openclaw/recipe-smoke-v1", "prompt": task["prompt"],
               "extra_info": {"tools_kwargs": {"task": task}}}
        with args.output.open("xb") as stream:
            pq.write_table(pa.Table.from_pylist([row]), stream)
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(task, stream, ensure_ascii=False, indent=2)
