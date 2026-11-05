# Diagnose a SWE-rebench Docker instance

When a manual test reports a missing module, first compare it with a gold-patch
run through the same Task and reward code used for evaluation. A direct `pytest`
command can omit the instance install, gold patch, test patch, or official test
targets. A package listed by `pip` can also have an editable installation that
points to a directory that no longer exists.

This guide uses the instance from [issue #267](https://github.com/verl-project/uni-agent/issues/267).
It diagnoses one public Docker image; it does not compare a veFaaS deployment or
explain the reporter's aggregate training reward.

## Prepare the inputs

Use Python 3.10–3.12, a running Docker daemon, and the repository checkout:

```bash
python -m pip install -e . pydantic pyyaml datasets swebench==4.1.0
python -m uni_agent.tasks.swe_rebench.preprocess --local-save-dir ./data

IMAGE=swerebench/sweb.eval.x86_64.smarie_1776_python-makefun-78@sha256:7c3cc14c4478ac81a6533da8fffb18e097dd747347dcccf1d5b853f05c0cbb2a
docker pull --platform linux/amd64 "$IMAGE"
```

Keep the preprocessed parquet and its checksum with the results. The checked
`nebius/SWE-rebench` revision was `89cdfbab4ab1bd8f5a658bb212d1b63624f4f881`;
its `data/filtered-00000-of-00001.parquet` SHA-256 was
`b2f037217a849732bf2e9b9d2b59d11561d03b4191a55c3142e1e58248817cfa`.
The preprocessor downloads the current dataset, so compare its instance metadata
before using newer data to reproduce these results.

The image is Linux AMD64. On Apple Silicon, configure AMD64 emulation in your
Docker VM and record the VM and emulator as part of the environment. The run
below used macOS ARM64, Colima's ARM64 VZ VM and Rosetta, Docker Engine 29.5.2,
Python 3.10.21 on the host, and Python 3.9.21 in the container. It was not a native
x86_64 or training-scale test.

## Check imports in the original image

```bash
docker run --rm --platform linux/amd64 --network none -w /testbed \
  --entrypoint bash "$IMAGE" -c '
    ls -ld /opt/miniconda3 /opt/conda /python-makefun /testbed
    /opt/conda/envs/testbed/bin/python -c "import importlib.util; print(importlib.util.find_spec(\"makefun\"))"
    /opt/conda/envs/testbed/bin/python -m pip show makefun
    cat /opt/conda/envs/testbed/lib/python3.9/site-packages/__editable__.makefun-1.12.1.pth
  '
```

For this digest, `/opt/miniconda3` is a symlink to `/opt/conda`, so changing
between these prefixes does not repair the import. `makefun` has installed
metadata, but its editable `.pth` points to `/python-makefun/src`; that directory
does not exist. The source is in `/testbed`. The reported command reproduces a
collection error with exit code 2:

```bash
docker run --rm --platform linux/amd64 --network none -w /testbed "$IMAGE" \
  /opt/conda/envs/testbed/bin/python -m pytest tests/test_generators_coroutines.py -v
```

## Run gold and no-patch controls

```bash
python examples/inference/diagnose_swe_rebench.py \
  --data-path ./data/swe_rebench_filtered.parquet \
  --instance-id smarie__python-makefun-78 \
  --image "$IMAGE" \
  --output-dir ./diagnostic-original
```

The example calls the inherited `SWEREBenchTask.run()` and the normal reward
function. Gold mode applies the dataset patch. The no-patch mode uses an agent
that makes no edits and never calls a model. Each runs in a fresh container with
2 CPUs and a 4 GiB memory limit. Docker networking remains enabled for the
instance installation. Ray, a Gateway, GPU training and real-model inference
are not exercised. This supplements the [parallel Oracle runner](oracle-verification.md).

The example requires a pre-pulled digest and creates a new output directory:

- `docker-version.txt` and `image-inspect.json` identify the runtime and image.
- Each control saves its Task config, generated `eval-script.sh`, complete
  `eval-output.log`, and `result.json`.
- Numbered command files preserve arguments, stdout, stderr and exit codes,
  including startup Python/package inspection and gold-patch application.
- `results.json` collects both completed controls. A Task exception is saved in
  `error.txt` and propagated; an unsolved reward is still a completed diagnostic.

Read the activation and installation output before the test summary. For
SWE-rebench, `eval_completed` reports the overall script exit status. The script
does not use `set -e`, and a successful final test-file reset can hide a failed
installation or pytest exit. `eval_completed=true` alone does not establish that
installation or tests passed. This example does not change that scoring behavior.

## Separate build-dependency drift from a solution failure

On 2026-11-05, the original per-instance `pip install -e .[dev]` failed with
`ModuleNotFoundError: No module named 'pkg_resources'` in its isolated build
environment; a verbose installation probe selected setuptools 82.0.1. This
project's `setup.py` imports `pkg_resources`, while its
build requirements leave setuptools unbounded above. Current setuptools
[removed `pkg_resources` in version 82](https://setuptools.pypa.io/en/stable/deprecated/pkg_resources.html).
Both original controls returned reward 0 with collection errors. These are
environment failures, not evidence that the gold patch is incorrect.

To test that hypothesis for this specific image, explicitly constrain its build
tools in a separate run:

```bash
cat > build-constraints.txt <<'EOF'
setuptools==75.8.0
setuptools-scm==8.2.0
wheel==0.45.1
EOF

python examples/inference/diagnose_swe_rebench.py \
  --data-path ./data/swe_rebench_filtered.parquet \
  --instance-id smarie__python-makefun-78 \
  --image "$IMAGE" \
  --build-constraints ./build-constraints.txt \
  --output-dir ./diagnostic-constrained
```

This explicitly adds `PIP_CONSTRAINT` inside each container and saves the file
with the results. It works with **pip 25.0 in this digest**; it is not a global
SWE-rebench workaround. Newer pip versions have different build-constraint
semantics; consult [pip's documentation](https://pip.pypa.io/en/stable/user_guide/#build-constraints)
before adapting it. The install still uses the network, and these three pins
are not a complete dependency lock.

Observed results with the same base commit, patches, test command and grader:

| Environment | Gold | No patch |
| --- | --- | --- |
| Original image and install | reward 0; collection errors | reward 0; collection errors |
| Same image with explicit build constraints | reward 1; 15 passed, 1 skipped | reward 0; 2 failed, 13 passed, 1 skipped |

The constrained install recreates the editable installation at `/testbed` and
generates the version module. The negative control fails exactly the two
`FAIL_TO_PASS` async-generator tests, while the gold control passes them.
The official targets are `src/makefun/_main_latest_py.py tests/_test_py36.py
tests/test_issues.py`, not the single file in the manual reproduction.
In a separate fresh container without either patch, the constrained install
also changed the original manual command from collection error to **4 passed**;
the imported module then came from `/testbed/src/makefun/__init__.py`.

These results establish a stale editable path in this public image and a
build-dependency incompatibility during this run. They do not establish the
reporter's historical build resolution, private provider image identity,
network policy, or the cause of low reward across the 1,150-instance training set.
Keep original and adjusted results separate when reporting an environment fix.
