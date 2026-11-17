# Evaluating troubleshooting skills

`evaluate.py` compares a new guide against two controls: existing file tools (`baseline`), and existing tools plus the same check scripts (`script`). The `skill` arm receives those scripts and the rest of the package, including `SKILL.md`. Script paths and contents are identical wherever provided. The runner measures a package available on disk; it does not assume native auto-discovery or give extra credit for opening the guide.

## Author and freeze a suite

Prepare a **private** suite directory outside the source tree:

```
suite/
  protocol.md             # decision rule, budgets, inclusion/exclusion, case provenance and dependencies
  package/                # copy of the proposed skill; no test answers
    SKILL.md
    scripts/
  cases/<case-key>/
    prompt.txt            # original ticket, or explicitly labelled reconstruction
    scene/                # full shareable, contemporaneous input, not excerpts
    rubric.json           # never staged in a participant's work directory
```

Use one key per independent incident, retain a source reference and snapshot, disclose shared sources, and report historical, controlled and smoke cases separately. A historical ticket without the original inputs is not a historical replay. If no eligible ticket exists, record zero cases and do not run the comparison. Write the primary metric and release criterion before trying the guide. Reserve previously unseen incidents; never tune on the test set then call it held-out. Independently confirm the answer key against the exhibits.

Example rubric **shape only** (invented names):

```json
{"kind":"smoke","provenance":"invented schema example","disposition":"ambiguous","candidates":["batch-a/item","batch-b/item"],"facts":{"request_id":null},"evidence":["a.json","b.json"],"next_step_rubric":"2: ask for the request ID or another discriminator available to the reporter; 1: ask for relevant but nonspecific logs; 0: guess one batch."}
```

`identified` requires one candidate; `ambiguous` requires at least two; `unknown` allows no candidate. An identifier includes its batch when identifiers repeat. Facts use typed JSON equality; missing and null are distinct. Evidence paths are relative to `scene/`; the required ones must be cited. Choose files that are essential regardless of the correct method. If alternate sources legitimately support a conclusion, do not require a particular path merely because the author used it. The `next_step_rubric` is case-specific and accepts equivalent actions. Source files, prompts, private rubrics, package contents, and the release rule are hashed together:

```
python evaluate.py freeze suite lock.json
python evaluate.py verify suite lock.json
```

## Run in fresh environments

`config.json` specifies the adapter command as an argument array, model identifier, integer seed, rounds, and wall time limit in seconds. Use an absolute executable path and absolute adapter path. Example shape:

```json
{"command":["/absolute/python","/absolute/your_isolated_adapter.py"],"model":"pinned-model-and-reasoning-setting","seed":0,"rounds":3,"timeout_seconds":360}
```

The adapter receives the ticket and a common response contract on stdin, runs once in a fresh directory, and returns **only one JSON object** on stdout. Put diagnostics on stderr. Configure model, reasoning, provider, tool budgets, and account on the adapter itself; `model` in config documents the setting, it does not configure an arbitrary executable. Pin its build. The adapter must expose only the staged files, with the same read/search tools for all arms, no external browsing, shared profile, host shell, git history, or global skill installation. **A temporary working directory alone is not an access boundary.** Use an isolated service/container or a tool-allowlisted adapter; run only reviewed skill scripts. Use sanitized evidence and inspect tool events before calling a trial blinded. The evaluator never copies credentials into participant files. Explicitly distinguish forced invocations from naturally triggered use and test off-topic prompts separately when evaluating discovery.

```
python evaluate.py run suite lock.json config.json runs-001
python evaluate.py replay suite runs-001
python evaluate.py review suite runs-001 reviewer-packet
python evaluate.py ratings suite runs-001 ratings.json
python evaluate.py summary suite runs-001 --votes ratings.json
```

Give an independent reviewer only `reviewer-packet/questions.json`; retain `review-key.json` with the experimenter. Reviewer returns `{ "<opaque id>": {"score":2,"reason":"asks for a discriminator"} }` for **every** row. Separate fact accuracy, false positive identification, and next action; an unanswered trial is an operational failure rather than a correct abstention. The blind packet hides arm labels and randomizes answer order; writing style or tool mentions may still reveal treatment. Where next-action grades decide release, have two reviewers score the same packet separately, retaining disagreements and their reasons.

Inspect `plan.json`, original stdout/stderr, input digests, receipts, and `results.json`. Replay uses the frozen suite and checks the evaluator version. Missing, invalid, failed and timed-out trials stay in the denominator; report their counts alongside wrong identifications so adapters which never answer cannot appear safe. A workspace file inventory establishes what was staged, not what a participant actually read. `seconds` is time until final submission (including startup), **not time to first correct evidence** or engineer hours saved. To claim earlier evidence, collect timestamped intermediate findings without correctness feedback and grade them against the same frozen rubric. Preserve the full transcript externally for audit. Report per-incident, per-arm counts and medians before aggregates. Three repetitions of one incident are still one incident; on tiny samples describe observed differences without claiming generalization. Do not mix smoke/controlled cases into historic incident coverage.

Run the offline harness checks with `python -B -m unittest discover -s evals/skills`. The tests generate temporary invented inputs solely to verify isolation of staged contents, failed-trial retention, replay and scoring; they supply **no evidence of skill effectiveness**.


## Why use these controls?

[SWE-Skills-Bench, October 2026 revision](https://arxiv.org/abs/2603.15401v2) compares requirements in paired environments with executable checks. It found no statistically significant effect for 92 of 98 skills; no improvement is a valid result. [The downstream utility study](https://arxiv.org/abs/2610.08875) measures the delta on the same tasks with a fixed model and harness. These are research methods, not an industry certification or a universal percentage threshold. This suite adds the scripts-only arm to separate document utility from tool availability. Freeze the release threshold and the number of incidents before examining answers. If evidence is scarce, publish descriptive results rather than repeating the same cases until a threshold is crossed.
