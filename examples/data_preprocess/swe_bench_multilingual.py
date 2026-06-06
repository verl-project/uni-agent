# ruff: noqa: E501
"""Preprocess SWE-bench/SWE-bench_Multilingual into the uni-agent SWE-agent format.

The dataset has 300 instances across 41 repos in 7 non-Python languages
(c/go/java/js/php/ruby/rust). Grading is the official ``swebench`` harness, wired up
by the ``swe_bench_multilingual`` reward spec, so we only keep the fields that
``make_test_spec`` + grading need: ``instance_id``, ``repo``, ``version``,
``base_commit``, ``test_patch``, ``patch`` (gold, for verifiers),
``problem_statement``, ``FAIL_TO_PASS``, ``PASS_TO_PASS``.

The repo lives at ``/testbed`` in the published ``swebench/sweb.eval.x86_64.<id>``
images. The image already holds the repo at ``base_commit`` plus the harness's
build-time ``pre_install``/``build`` edits (e.g. apache/lucene injects a gradle
``testLogging`` block the parser relies on). Those edits are uncommitted, so
post-setup *commits* them (instead of ``git reset --hard``, which would revert them
and break grading) -- giving the agent a clean tree while preserving the build
config.

Example::

    DEPLOYMENT=modal python examples/data_preprocess/swe_bench_multilingual.py \
        --local-save-dir ~/data/swe_agent
"""

import argparse
import os

from datasets import load_dataset
from swebench.harness.constants import MAP_REPO_TO_EXT

impl = os.getenv("DEPLOYMENT", "modal").lower()
if impl != "modal":
    # Only the public ``swebench/`` Docker Hub images cover the multilingual set.
    raise ValueError("SWE-bench_Multilingual preprocessing only supports modal deployment")


def get_image_name(instance_id: str) -> str:
    """Published image ref, mirroring swebench's ``instance_image_key``."""
    return f"swebench/sweb.eval.x86_64.{instance_id.lower().replace('__', '_1776_')}"


# Map swebench file-extension code -> human language name for the prompt.
EXT_TO_LANGUAGE = {
    "c": "C",
    "go": "Go",
    "java": "Java",
    "js": "JavaScript",
    "php": "PHP",
    "rb": "Ruby",
    "rs": "Rust",
}


SYSTEM_PROMPT = """
You are a helpful assistant that can interact with a computer to solve tasks.
""".strip()

USER_PROMPT = """
<uploaded_files>
/testbed
</uploaded_files>
I have uploaded a code repository in the /testbed directory (primary language: {language}). You can explore and modify files using the available tools. Consider the following issue description:

<issue_description>
{problem_statement}
</issue_description>

Can you help me implement the necessary changes to the repository to fix the <issue_description>?
I have already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Also the development environment is already set up for you (i.e., all dependencies are already installed and the project is already built), so you don't need to install other packages.
Your task is to make the minimal changes to non-test files in the /testbed directory to ensure the <issue_description> is satisfied.

Follow these steps to resolve the issue:
1. First, explore the codebase to locate and understand the code relevant to the <issue_description>.
- Use efficient search commands to identify key files and functions.
- Build your understanding of how the code works, the expected behaviors and edge cases, and the potential root causes for the given issue.

2. Assess whether you can reproduce the issue:
- Create a small script (e.g. at '/testbed/reproduce_issue.*') that demonstrates the error, using the repository's own language/runtime.
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


def build_swe_bench_multilingual():
    def process(example):
        repo = example["repo"]
        instance_id = example["instance_id"]
        language = EXT_TO_LANGUAGE.get(MAP_REPO_TO_EXT[repo], "the project's")

        metadata = {
            "instance_id": instance_id,
            "repo": repo,
            "version": str(example["version"]),
            "base_commit": example["base_commit"],
            "patch": example["patch"],
            "test_patch": example["test_patch"],
            "problem_statement": example["problem_statement"],
            "FAIL_TO_PASS": example["FAIL_TO_PASS"],
            "PASS_TO_PASS": example["PASS_TO_PASS"],
        }

        reset_script = " && ".join(
            [
                "git tag -d $(git tag -l)",
                "git reflog expire --expire=now --all",
                "git gc --prune=now",
            ]
        )

        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT.format(
                        language=language,
                        problem_statement=example["problem_statement"],
                    ),
                },
            ],
            "agent_name": "swe_agent",
            "extra_info": {
                "tools_kwargs": {
                    "env": {
                        "deployment": {
                            "image": get_image_name(instance_id),
                            "modal_sandbox_kwargs": {
                                "cpu": (0.5, 4.0),
                                "memory": (1024, 8192),
                            },
                        },
                        "post_setup_cmd": reset_script,
                    },
                    "reward": {
                        "name": "swe_bench_multilingual",
                        "metadata": metadata,
                    },
                },
            },
        }

    data_source = "SWE-bench/SWE-bench_Multilingual"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="test")
    print(f"Loaded {len(dataset)} raw instances", flush=True)

    # Test set: keep every instance (no filtering).
    dataset = dataset.map(process, remove_columns=dataset.column_names)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-save-dir", default="~/data/swe_agent")
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    sbm_dataset = build_swe_bench_multilingual()
    out_path = f"{save_dir}/swe_bench_multilingual_{impl}.parquet"
    sbm_dataset.to_parquet(out_path)
    print(f"Wrote {len(sbm_dataset)} instances to {out_path}", flush=True)
