# ruff: noqa: E501
"""Preprocess the nebius/SWE-rebench-V2 dataset into the uni-agent SWE-agent format.

Aligned with the official evaluator
(https://github.com/SWE-rebench/SWE-rebench-V2, ``scripts/eval.py`` +
``combine.Dockerfile.j2``). The matching reward spec is
``uni_agent/reward/swe_rebench_v2.py`` (registered as ``swe_rebench_v2``).

What's KEPT (the reward needs these): ``instance_id``, ``repo``, ``base_commit``,
``patch`` (gold, for golden-eval / verifiers), ``test_patch``, ``problem_statement``,
``FAIL_TO_PASS``, ``PASS_TO_PASS``, and from ``install_config`` only ``test_cmd``
and ``log_parser``. We also derive ``project_dir`` (= ``/<repo_name>``) and the
per-instance ``image_name``.

What's DROPPED: ``pr_description``, ``interface``, ``license``, ``created_at``,
``meta`` (only used for optional filtering), and ``install`` — the install
commands already ran at *image build time* (see ``combine.Dockerfile.j2``), so the
reward must NOT re-run them. There is also no ``FAIL_TO_FAIL`` / ``PASS_TO_FAIL``
in V2.

Key DIFFERENCES vs. SWE-bench / SWE-rebench-v1 preprocessing:
* The repo lives at ``/<repo_name>`` (e.g. ``/netcdf-c``), not ``/testbed`` —
  the prompt and the post-setup reset target that directory.
* V2 is language-agnostic; the prompt is generalized and tests are graded by the
  per-instance ``log_parser`` (any of the 76 vendored parsers).
* The post-setup reset does NOT ``git clean`` — that would delete the compiled
  artifacts produced during image build and break many non-Python test suites.

Example::

    DEPLOYMENT=modal python examples/data_preprocess/swe_rebench_v2.py \
        --local-save-dir ~/data/swe_agent --language python --max-samples 5000
"""
import argparse
import os

from datasets import load_dataset

from uni_agent.reward._swe_rebench_v2 import NAME_TO_PARSER

impl = os.getenv("DEPLOYMENT", "modal").lower()
if impl == "local":
    raise NotImplementedError("Local deployment is not implemented yet")
elif impl not in ("modal", "vefaas"):
    raise ValueError(f"Invalid deployment implementation: {impl}")

# V2 images are published under ``docker.io/swerebenchv2/``. ``modal`` pulls them
# directly. To use a private mirror (e.g. a veFaaS container registry that mirrors
# the same ``<repo>:<tag>``), set SWEREBENCH_V2_REGISTRY, e.g.
#   SWEREBENCH_V2_REGISTRY=enterprise-public-2-cn-beijing.cr.volces.com/swe-rebench-v2
_REGISTRY_OVERRIDE = os.getenv("SWEREBENCH_V2_REGISTRY", "").rstrip("/")

# Parsers our vendored reward can actually grade with.
SUPPORTED_PARSERS = set(NAME_TO_PARSER.keys())


def resolve_image_name(image_name: str) -> str:
    """Return the image reference to deploy, applying an optional registry swap."""
    if not image_name:
        raise ValueError("SWE-rebench-V2 instance is missing `image_name`")
    if not _REGISTRY_OVERRIDE:
        return image_name
    # image_name looks like "docker.io/swerebenchv2/<repo>:<tag>"; keep "<repo>:<tag>".
    repo_tag = image_name.split("/", maxsplit=2)[-1]
    return f"{_REGISTRY_OVERRIDE}/{repo_tag}"


def project_dir_for(repo: str) -> str:
    """Working directory inside the instance image (mirrors combine.Dockerfile.j2)."""
    return f"/{repo.split('/')[1]}"


SYSTEM_PROMPT = """
You are a helpful assistant that can interact with a computer to solve tasks.
""".strip()

USER_PROMPT = """
<uploaded_files>
{project_dir}
</uploaded_files>
I have uploaded a code repository in the {project_dir} directory (primary language: {language}). You can explore and modify files using the available tools. Consider the following issue description:

<issue_description>
{problem_statement}
</issue_description>

Can you help me implement the necessary changes to the repository to fix the <issue_description>?
I have already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Also the development environment is already set up for you (i.e., all dependencies are already installed and the project is already built), so you don't need to install other packages.
Your task is to make the minimal changes to non-test files in the {project_dir} directory to ensure the <issue_description> is satisfied.

Follow these steps to resolve the issue:
1. First, explore the codebase to locate and understand the code relevant to the <issue_description>.
- Use efficient search commands to identify key files and functions.
- You should err on the side of caution and look at various relevant files and build your understanding of
    - how the code works
    - what are the expected behaviors and edge cases
    - what are the potential root causes for the given issue

2. Assess whether you can reproduce the issue:
- Create a small script (e.g. at '{project_dir}/reproduce_issue.*') that demonstrates the error, using the repository's own language/runtime.
- Execute this script to confirm the error behavior.
- You should reproduce the issue before fixing it.
- Your reproduction script should also assert the expected behavior for the fixed code.

3. Analyze the root cause:
- Identify the underlying problem based on your code exploration and reproduction results.
- Critically analyze different potential approaches to fix the issue.
- You NEED to explicitly reason about multiple approaches to fix the issue. Next, find the most elegant and effective solution among them considering the tradeoffs (correctness, generality, side effects, etc.).
- You would need to reason about execution paths, edge cases, and other potential issues. You should look at the unit tests to understand the expected behavior of the relevant code.

4. Implement your solution:
- Make targeted changes to the necessary files following idiomatic code patterns once you determine the root cause.
- You should be thorough and methodical.

5. Verify your solution:
- Rerun your reproduction script to confirm the error is fixed.
- If verification fails, iterate on your solution until successful. If you identify the reproduction script is buggy, adjust it as needed.

6. Run unit tests:
- Find and run the relevant unit tests relevant to the performed fix using the project's own test runner.
- You should run the unit tests to ensure your solution is correct and does not cause any regressions.
- In cases where the unit tests do not pass, you should consider whether the unit tests do not reflect the *new* expected behavior of the code. If so, you can test it by writing additional edge test cases.
- RUN ALL relevant unit tests to ensure your solution is correct and does not cause any regressions.
- DO NOT MODIFY any of the existing unit tests. You can add new edge test cases in a separate file if needed BUT DO NOT MODIFY THE EXISTING TESTS.

7. Test edge cases:
- Identify potential edge cases that might challenge your solution.
- Create additional test cases in a separate file.
- Execute these tests to verify your solution's robustness.
- You should run multiple rounds of edge cases. When creating edge cases:
    - Consider complex scenarios beyond the original issue description
    - Test for regressions to ensure existing functionality remains intact
    - At each round you should write multiple edge test cases in the same file to be efficient

8. Refine if necessary:
- If edge case testing reveals issues, refine your solution accordingly.
- Ensure your final implementation handles all identified scenarios correctly.
- Document any assumptions or limitations of your solution.

9. Submit your solution:
- Once you have verified your solution, submit your solution using the `submit` tool.

A successful resolution means:
- The specific error/issue described no longer occurs
- Your changes maintain compatibility with existing functionality
- Edge cases are properly handled
""".strip()


def _llm_metadata(example: dict) -> dict:
    meta = example.get("meta") or {}
    llm = meta.get("llm_metadata")
    # V2 stores llm_metadata as a list of dicts (sometimes a single dict).
    if isinstance(llm, list):
        return llm[0] if llm and isinstance(llm[0], dict) else {}
    return llm if isinstance(llm, dict) else {}


def _make_keep_fn(args):
    languages = None if args.language.lower() == "all" else {x.strip().lower() for x in args.language.split(",") if x.strip()}
    if args.parsers.lower() == "all":
        parsers = None
    elif args.parsers.lower() == "supported":
        parsers = SUPPORTED_PARSERS
    else:
        parsers = {x.strip() for x in args.parsers.split(",") if x.strip()}
    difficulties = None if not args.difficulty else {x.strip().lower() for x in args.difficulty.split(",") if x.strip()}

    def keep(example: dict) -> bool:
        if languages is not None and (example.get("language") or "").lower() not in languages:
            return False
        install_config = example.get("install_config") or {}
        log_parser = install_config.get("log_parser")
        # Always require a parser we can actually grade with.
        if not log_parser or log_parser not in SUPPORTED_PARSERS:
            return False
        if parsers is not None and log_parser not in parsers:
            return False
        if not example.get("repo") or "/" not in example["repo"]:
            return False
        if not example.get("image_name"):
            return False
        if not install_config.get("test_cmd"):
            return False
        if not (example.get("test_patch") or "").strip():
            return False
        if args.drop_empty_problem_statement and not (example.get("problem_statement") or "").strip():
            return False
        llm_meta = _llm_metadata(example)
        if args.min_confidence is not None:
            confidence = llm_meta.get("confidence")
            if confidence is None or confidence < args.min_confidence:
                return False
        if difficulties is not None and (llm_meta.get("difficulty") or "").lower() not in difficulties:
            return False
        return True

    return keep


def build_swe_rebench_v2(args):
    def process(example):
        install_config = example["install_config"]
        repo = example["repo"]
        project_dir = project_dir_for(repo)

        # Lean metadata: exactly what the swe_rebench_v2 reward consumes.
        metadata = {
            "instance_id": example["instance_id"],
            "repo": repo,
            "project_dir": project_dir,
            "base_commit": example["base_commit"],
            "patch": example["patch"],
            "test_patch": example["test_patch"],
            "problem_statement": example["problem_statement"],
            "language": example["language"],
            "FAIL_TO_PASS": example["FAIL_TO_PASS"],
            "PASS_TO_PASS": example["PASS_TO_PASS"],
            "test_cmd": install_config["test_cmd"],  # list[str], run as-is
            "log_parser": install_config["log_parser"],
        }
        image_name = resolve_image_name(example["image_name"])

        # The instance image already has the repo at base_commit AND build
        # artifacts from the image-build install step. Reset tracked files to the
        # base revision as a safety net but do NOT `git clean` (that would delete
        # the compiled artifacts and break many test suites).
        reset_cmds = [
            f"cd {project_dir} || true",
            f"git config --global --add safe.directory {project_dir} || true",
            f"git reset --hard {example['base_commit']} || true",
        ]
        reset_script = " && ".join(reset_cmds)

        sample = {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT.format(
                        project_dir=project_dir,
                        language=example["language"],
                        problem_statement=example["problem_statement"],
                    ),
                },
            ],
            "agent_name": "swe_agent",
            "extra_info": {
                "tools_kwargs": {
                    "env": {
                        "deployment": {"image": image_name},
                        "post_setup_cmd": reset_script,
                    },
                    "reward": {
                        "name": "swe_rebench_v2",
                        "metadata": metadata,
                    },
                },
            },
        }
        return sample

    data_source = "nebius/SWE-rebench-V2"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="train")
    print(f"Loaded {len(dataset)} raw instances", flush=True)

    dataset = dataset.filter(_make_keep_fn(args))
    print(f"{len(dataset)} instances remain after filtering", flush=True)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        if args.shuffle_seed is not None:
            dataset = dataset.shuffle(seed=args.shuffle_seed)
        dataset = dataset.select(range(args.max_samples))
        print(f"Truncated to {len(dataset)} instances (max_samples={args.max_samples})", flush=True)

    dataset = dataset.map(process, remove_columns=dataset.column_names)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-save-dir", default="~/data/swe_agent")
    parser.add_argument(
        "--language",
        default="all",
        help="Comma-separated language filter (V2 codes, e.g. python,go,ts), or 'all'.",
    )
    parser.add_argument(
        "--parsers",
        default="supported",
        help="'supported' (all parsers the reward implements), 'all' (alias of supported), "
        "or a comma-separated allowlist. Instances with an unsupported parser are always dropped.",
    )
    parser.add_argument(
        "--difficulty",
        default="",
        help="Optional comma-separated meta.llm_metadata.difficulty filter (e.g. easy,medium).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Optional minimum meta.llm_metadata.confidence (0-1).",
    )
    parser.add_argument(
        "--no-drop-empty-problem-statement",
        dest="drop_empty_problem_statement",
        action="store_false",
        help="Keep instances whose problem_statement is empty (dropped by default).",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Cap the number of output instances.")
    parser.add_argument("--shuffle-seed", type=int, default=None, help="Shuffle seed used before --max-samples.")
    parser.set_defaults(drop_empty_problem_statement=True)

    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    sbv2_dataset = build_swe_rebench_v2(args)
    out_path = f"{save_dir}/swe_rebench_v2_{impl}.parquet"
    sbv2_dataset.to_parquet(out_path)
    print(f"Wrote {len(sbv2_dataset)} instances to {out_path}", flush=True)
