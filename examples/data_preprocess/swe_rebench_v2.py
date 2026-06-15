# ruff: noqa: E501
"""Preprocess the Python slice of ``nebius/SWE-rebench-V2`` into the uni-agent SWE-agent format.

SWE-rebench-V2 is a language-agnostic SWE task collection (32,079 instances, 20
languages) derived from real GitHub issues/PRs. **We currently keep only the
Python slice** (7,243 instances, all graded with ``parse_log_pytest``); the
machinery below stays language-aware so other languages can be re-enabled later
(pass ``--languages`` and register their parser in
``uni_agent/reward/swe_rebench_v2_log_parsers.py``).

Unlike SWE-bench it is graded by the dataset authors' own per-framework log
parsers (vendored in ``uni_agent/reward/swe_rebench_v2_log_parsers.py``) rather
than the ``swebench`` harness, and each instance ships:

* a prebuilt instance image in the top-level ``image_name`` field
  (e.g. ``docker.io/swerebenchv2/wtforms-wtforms:614-848d28d``), and
* an ``install_config`` carrying ``install`` (run at image build time),
  ``test_cmd`` and ``log_parser`` (used at grading time).

The repo is checked out at ``/<repo_name>`` (e.g. ``/wtforms`` for
``wtforms/wtforms``), mirroring the upstream ``combine.Dockerfile.j2`` which
clones into ``/{repo.split('/')[1]}`` -- *not* ``/testbed``. We therefore keep
only the fields the ``swe_rebench_v2`` reward spec needs and point the prompt at
the right directory.

Examples::

    # Python slice (default), uses the dataset's docker.io images directly:
    python examples/data_preprocess/swe_rebench_v2.py --local-save-dir ~/data/swe_agent

    # A capped Python subset:
    python examples/data_preprocess/swe_rebench_v2.py --max-instances 500
"""

import argparse
import os

from datasets import load_dataset

# Sandbox backend: affects the output filename and (for modal) the resource cap.
impl = os.getenv("DEPLOYMENT", "modal").lower()

# Image names ship in the dataset (docker.io/swerebenchv2/<name>:<tag>). If you
# mirror them to another registry, set SRB2_IMAGE_REGISTRY to that prefix and the
# leading ``docker.io/swerebenchv2`` is swapped for it (tag preserved).
_DATASET_IMAGE_PREFIX = "docker.io/swerebenchv2/"


def resolve_image_name(image_name: str) -> str:
    registry = os.getenv("SRB2_IMAGE_REGISTRY", "").strip().rstrip("/")
    if not registry:
        return image_name
    name = image_name.split("/")[-1] if image_name.startswith(_DATASET_IMAGE_PREFIX) else image_name
    return f"{registry}/{name}"


# Human-readable language names for the prompt (dataset ``language`` codes).
# Python-only for now; add the other dataset codes here to enable them.
LANGUAGE_NAMES = {
    "python": "Python",
}
SKIP_SAMPLES = []

SYSTEM_PROMPT = """
You are a helpful assistant that can interact with a computer to solve tasks.
""".strip()

USER_PROMPT = """
<uploaded_files>
{repo_dir}
</uploaded_files>
I have uploaded a code repository in the {repo_dir} directory (primary language: {language}). You can explore and modify files using the available tools. Consider the following issue description:

<issue_description>
{problem_statement}
</issue_description>

Can you help me implement the necessary changes to the repository to fix the <issue_description>?
I have already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Also the development environment is already set up for you (i.e., all dependencies are already installed and the project is already built), so you don't need to install other packages.
Your task is to make the minimal changes to non-test files in the {repo_dir} directory to ensure the <issue_description> is satisfied.

Follow these steps to resolve the issue:
1. First, explore the codebase to locate and understand the code relevant to the <issue_description>.
- Use efficient search commands to identify key files and functions.
- Build your understanding of how the code works, the expected behaviors and edge cases, and the potential root causes for the given issue.

2. Assess whether you can reproduce the issue:
- Create a small script (e.g. at '{repo_dir}/reproduce_issue.*') that demonstrates the error, using the repository's own language/runtime.
- Execute this script to confirm the error behavior before fixing it.
- Your reproduction script should also assert the expected behavior for the fixed code.

3. Analyze the root cause:
- Identify the underlying problem based on your code exploration and reproduction results.
- Reason about multiple potential approaches and pick the most elegant and effective one, considering correctness, generality, and side effects.

4. Implement your solution:
- Make targeted changes to the necessary files following idiomatic code patterns once you determine the root cause.

5. Verify your solution:
- Rerun your reproduction script to confirm the error is fixed, iterating until successful.

6. Run unit tests:
- Find and run the relevant unit tests using the project's own test runner to ensure your solution is correct and does not cause regressions.
- DO NOT MODIFY any of the existing unit tests. You can add new edge test cases in a separate file if needed BUT DO NOT MODIFY THE EXISTING TESTS.

7. Test edge cases:
- Identify potential edge cases that might challenge your solution, create additional tests in a separate file, and verify robustness.

8. Submit your solution:
- Once you have verified your solution, submit it using the `submit` tool.

A successful resolution means:
- The specific error/issue described no longer occurs
- Your changes maintain compatibility with existing functionality
- Edge cases are properly handled
""".strip()


def build_swe_rebench_v2(languages: set[str] | None, max_instances: int | None):
    def process(example):
        repo = example["repo"]
        instance_id = example["instance_id"]
        repo_dir = f"/{repo.split('/')[1]}"
        lang_code = (example.get("language") or "").lower()
        language = LANGUAGE_NAMES.get(lang_code, lang_code or "the project's")

        install_config = example["install_config"] or {}

        metadata = {
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": example["base_commit"],
            "patch": example["patch"],
            "test_patch": example["test_patch"],
            "problem_statement": example["problem_statement"],
            "language": lang_code,
            "FAIL_TO_PASS": list(example["FAIL_TO_PASS"]),
            "PASS_TO_PASS": list(example["PASS_TO_PASS"]),
            # Grading inputs (used by the swe_rebench_v2 reward spec).
            "log_parser": install_config["log_parser"],
            "test_cmd": install_config["test_cmd"],
        }

        reset_script = " && ".join(
            [
                "git tag -d $(git tag -l) 2>/dev/null || true",
                "git reflog expire --expire=now --all 2>/dev/null || true",
                "git gc --prune=now 2>/dev/null || true",
            ]
        )

        deployment = {"image": resolve_image_name(example["image_name"])}
        if impl == "modal":
            deployment["modal_sandbox_kwargs"] = {"cpu": (0.5, 4.0), "memory": (1024, 8192)}

        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT.format(
                        repo_dir=repo_dir,
                        language=language,
                        problem_statement=example["problem_statement"],
                    ),
                },
            ],
            "agent_name": "swe_agent",
            "extra_info": {
                "tools_kwargs": {
                    "env": {
                        "deployment": deployment,
                        "post_setup_cmd": reset_script,
                    },
                    "reward": {
                        "name": "swe_rebench_v2",
                        "metadata": metadata,
                    },
                },
            },
        }

    data_source = "nebius/SWE-rebench-V2"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="train")
    print(f"Loaded {len(dataset)} raw instances", flush=True)

    if languages:
        wanted = {lang.lower() for lang in languages}
        dataset = dataset.filter(lambda ex: (ex.get("language") or "").lower() in wanted)
    dataset = dataset.filter(lambda ex: ex["instance_id"] not in SKIP_SAMPLES)

    print(f"Kept {len(dataset)} instances after language filter {sorted(wanted)}", flush=True)

    if max_instances is not None and max_instances >= 0:
        dataset = dataset.select(range(min(max_instances, len(dataset))))
        print(f"Capped to {len(dataset)} instances", flush=True)

    dataset = dataset.map(process, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda ex: ex["extra_info"]["tools_kwargs"]["env"]["deployment"]["image"] is not None)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-save-dir", default="~/data/swe_agent")
    parser.add_argument(
        "--languages",
        default="python",
        help="Comma-separated language filter (dataset codes). Defaults to 'python'; "
        "pass '' to keep all languages (also register their parsers first).",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Optional cap on the number of instances kept (after filtering).",
    )
    args = parser.parse_args()

    languages = {x.strip() for x in args.languages.split(",") if x.strip()} or None

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    dataset = build_swe_rebench_v2(languages=languages, max_instances=args.max_instances)
    out_path = f"{save_dir}/swe_rebench_v2_{impl}.parquet"
    dataset.to_parquet(out_path)
    print(f"Wrote {len(dataset)} instances to {out_path}", flush=True)
