"""AI-to-agent GSM8K homework grading runner (TA role).

The external LLM plays a TA who uses the agent under test to grade student
homework. The homework files (homework1/i.txt) are assumed to already exist in
the workspace (copied from a prior student_chat.py run). The TA provides the
original question and ground truth answer, asks the agent to grade, reviews the
feedback, and then has it appended to the file.

The agent under test is reached through ``eval_common.AgentEndpoint`` (gateway or
OpenAI-compatible) -- see ``eval_common`` for environment configuration.

Usage:
  python TA_chat.py --dataset GSM8K.json --num-problems 5
"""

import argparse
import os

from eval_common import (
    AgentEndpoint,
    ensure_homework_dir,
    get_external_model,
    get_workspace,
    load_dataset,
    make_external_client,
    run_role_session,
)

HOMEWORK_DIR = "homework1"
SOURCE_HOMEWORK_DIR = "homework"

TA_SYSTEM_PROMPT = """\
You are role-playing as a TA who is grading student homework. You talk casually. \
You want the AI to write comments that are detailed and specific.

You CANNOT grade, rewrite, rephrase, or produce any comments yourself. \
You can ONLY tell the AI what to do. Never do the grading yourself. \
You know NOTHING about the homework content — you have not read it and \
you do not know any numbers, steps, or answers from the problem. \
Only the AI can read the file and see what the student wrote. \
You must NEVER mention any specific numbers, calculations, formulas, \
problem details, or student answers in your messages. \
You must NEVER list steps the student should have taken. \
You must NEVER write example comments or sample feedback. \
You must NEVER do any part of the grading yourself. \
Your ONLY job is to give short, general instructions to the AI — \
things like "make it more detailed", "mention whether the student \
showed their work", "check if the student handled all parts of the \
problem". Keep your instructions general and brief.

Your goal: get the AI to grade the student's homework and write comments. \
The comments must be detailed and specific. If the AI's comments are too \
short or not specific enough, tell it to rewrite using short, general phrases. \
Examples of good things to say: \
"Write more specific grading comments." \
"Make your grading more detailed — go deeper into the student's reasoning." \
"Your comments are too short, please expand them." \
"Be more specific about what the student got right and wrong." \
"Add more detail about the steps the student took." \
Just tell it to fix it — don't fix it yourself. \
If the comments are already detailed and specific, no need to rewrite.

Steps:
1. Look at what the AI gives you. If the comments are not detailed or specific enough, \
tell it to redo it in short, general terms (e.g. "make your grading more specific" \
or "add more detail about the student's reasoning" — NOT "the student should have \
done X then Y"). \
Do NOT mention writing to the file in the same message. Only ask for a rewrite.
2. After the AI shows you the satisfactory version, THEN in a \
separate message ask it to append the comments to the end of the homework file \
(not overwrite it). Do NOT combine a rewrite request and a write request.
3. After the AI says it saved the file, say exactly: GRADING_DONE

Never say GRADING_DONE until the AI confirms it wrote the file.
Never write or grade anything yourself. Never reference specific problem content. \
Just give short, general instructions."""

FIRST_MESSAGE_TEMPLATE = """\
I'm grading a student's homework. The submission is in the file homework1/{index}.txt \
in your workspace. Please read the file first.

Here is the original question and the correct answer for reference:

Question: {question}

Correct answer: {ground_truth}

Please read the student's submission from the file, compare it with the correct \
answer, and write the grading comments directly. No intro, no summary, no \
"here are the comments" — just the comments themselves, as if you are writing \
them on the student's paper. \
Show me the comments first — don't write to the file until I tell you to."""

DONE_SENTINEL = "GRADING_DONE"


def main():
    parser = argparse.ArgumentParser(description="GSM8K homework grading runner via the agent under test (TA role)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to GSM8K.json")
    parser.add_argument("--num-problems", type=int, default=5, help="Number of problems to grade (default: 5)")
    parser.add_argument("--max-turns", type=int, default=8, help="Max turns per problem (default: 8)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per network call (default: 3)")
    parser.add_argument("--output", type=str, default="results_TA.txt", help="Output file for agent replies")
    args = parser.parse_args()

    endpoint = AgentEndpoint.from_env()
    external_client = make_external_client()
    model = get_external_model()
    workspace = get_workspace()

    ensure_homework_dir(workspace, SOURCE_HOMEWORK_DIR, HOMEWORK_DIR)

    problems = load_dataset(args.dataset)
    count = min(args.num_problems, len(problems))
    print(f"Loaded {len(problems)} problems from {args.dataset}")
    print(f"Grading {count} problems, max {args.max_turns} turns each | endpoint mode: {endpoint.mode}\n")

    open(args.output, "w").close()
    print(f"Output file: {args.output} (cleared)\n")

    results = []
    for i in range(count):
        question = problems[i].get("question", "")
        ground_truth = problems[i].get("ground_truth_answer", "?")
        completed = run_role_session(
            session_id=f"ta-grade-{i}-{os.getpid()}",
            endpoint=endpoint,
            external_client=external_client,
            model=model,
            system_prompt=TA_SYSTEM_PROMPT.replace("homework file", f"file {HOMEWORK_DIR}/{i}.txt"),
            first_message=FIRST_MESSAGE_TEMPLATE.format(index=i, question=question, ground_truth=ground_truth),
            done_sentinel=DONE_SENTINEL,
            role_label=f"Grading problem {i} (ground truth: {ground_truth})",
            max_turns=args.max_turns,
            max_retries=args.max_retries,
            output_file=args.output,
        )
        results.append(completed)

    done = sum(results)
    print(f"\n{'#' * 60}")
    print(f"# Summary: {done}/{count} problems graded within turn limit")
    print(f"{'#' * 60}")
    for i, ok in enumerate(results):
        status = "done" if ok else "incomplete"
        gt = problems[i].get("ground_truth_answer", "?")
        print(f"  Problem {i}: {status} (ground truth: {gt})")


if __name__ == "__main__":
    main()
