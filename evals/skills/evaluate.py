"""Run a frozen paired skill evaluation. Python 3.10+, standard library only."""

import argparse
import hashlib
import json
import os
import random
import secrets
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath, PureWindowsPath

ARMS = ("baseline", "script", "skill")
STATES = ("identified", "ambiguous", "unknown")
KINDS = ("historical", "controlled", "smoke")
RESPONSE = (
    "Return one JSON object with disposition (identified, ambiguous, unknown), "
    "candidates (list of complete candidate identifiers), facts (object), evidence "
    "(list of relative paths under scene), and next_step (specific action or reason "
    "no further action is needed). Mark missing facts null. Do not assume a unique "
    "candidate unless the evidence distinguishes it."
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(root):
    result = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink() or bool(getattr(p.lstat(), "st_file_attributes", 0) & 1024):
            raise ValueError(f"Links are not portable evidence: {p}")
        if ".git" in p.parts or "__pycache__" in p.parts:
            raise ValueError(f"Unexpected repository or cache: {p}")
        if p.is_file():
            result[p.relative_to(root).as_posix()] = digest(p)
    return result


MAX_ARTIFACT_BYTES = 4 * 1024 * 1024
JSON_TYPES = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "object": (dict,),
    "array": (list,),
    "boolean": (bool,),
    "null": (type(None),),
}


def artifact_rules(spec):
    rules = spec.get("outputs")
    if not isinstance(rules, dict) or not rules:
        raise ValueError("Artifact profile requires private expected JSON outputs")
    for name in rules:
        path = PurePosixPath(name)
        if (
            not path.parts
            or path.as_posix() != name
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or ":" in name
            or PureWindowsPath(name).drive
            or any(PureWindowsPath(part).is_reserved() for part in path.parts)
        ):
            raise ValueError("Output names must be portable relative paths under output/")
    return rules


def public_contract(case, spec):
    if spec.get("profile") == "artifact":
        return (
            "Write the requested JSON files under output/: "
            + json.dumps(sorted(artifact_rules(spec)))
            + ". Return one JSON object on stdout ({} is sufficient); the evaluator grades the files themselves."
        )
    contract = load(case / "answer-schema.json")
    types = contract.get("fact_types")
    if (
        not isinstance(types, dict)
        or set(types) != set(spec["facts"])
        or any(not isinstance(t, str) or t not in JSON_TYPES for t in types.values())
        or any(v is not None and type(v) not in JSON_TYPES[types[k]] for k, v in spec["facts"].items())
        or not isinstance(contract.get("candidate_format"), str)
        or not contract["candidate_format"].strip()
    ):
        raise ValueError(f"Invalid public fact types or candidate format: {case.name}")
    return (
        RESPONSE
        + " Use exactly these fact fields and JSON types (null when unknown): "
        + json.dumps(types, sort_keys=True)
        + ". Candidate format: "
        + contract["candidate_format"]
    )


def cases(root):
    answer = []
    for case in sorted((root / "cases").iterdir()):
        if not case.is_dir():
            raise ValueError(f"Unexpected file among cases: {case}")
        spec = load(case / "rubric.json")
        if spec["kind"] not in KINDS or not spec["provenance"].strip():
            raise ValueError(f"Invalid provenance: {case.name}")
        if not inventory(case / "scene") or not (case / "prompt.txt").read_text(encoding="utf-8").strip():
            raise ValueError(f"Case needs original inputs and a task: {case.name}")
        if spec.get("profile") == "artifact":
            artifact_rules(spec)
            answer.append(case.name)
            continue
        public_contract(case, spec)
        if spec.get("profile", "diagnosis") != "diagnosis" or spec["disposition"] not in STATES:
            raise ValueError(f"Invalid provenance kind or disposition: {case.name}")
        ids = spec["candidates"]
        if not isinstance(ids, list) or not all(isinstance(x, str) and x for x in ids) or len(ids) != len(set(ids)):
            raise ValueError("candidates must be unique nonempty strings")
        if (spec["disposition"] == "identified" and len(ids) != 1) or (
            spec["disposition"] == "ambiguous" and len(ids) < 2
        ):
            raise ValueError("Disposition incompatible with expected candidates")
        if not spec["next_step_rubric"].strip() or not isinstance(spec["facts"], dict):
            raise ValueError(f"Missing provenance, grading rubric, or facts: {case.name}")
        paths = set(inventory(case / "scene"))
        if not isinstance(spec["evidence"], list) or not spec["evidence"] or not set(spec["evidence"]) <= paths:
            raise ValueError(f"Missing input or evidence path: {case.name}")
        answer.append(case.name)
    if (
        not answer
        or not (root / "package/SKILL.md").is_file()
        or not (root / "protocol.md").read_text(encoding="utf-8").strip()
    ):
        raise ValueError("Suite requires cases, a protocol and SKILL.md")
    return answer


def frozen(root):
    files = inventory(root)
    arms = ARMS if inventory(root / "package/scripts") else ("baseline", "skill")
    return {"version": 2, "cases": cases(root), "arms": list(arms), "sha256": files}


def score(spec, answer, scene_paths):
    if not isinstance(answer, dict):
        answer = {}
    ids = answer.get("candidates")
    state = answer.get("disposition")
    facts = answer.get("facts")
    paths = answer.get("evidence")
    valid_ids = isinstance(ids, list) and all(isinstance(x, str) for x in ids) and len(ids) == len(set(ids))
    facts = facts if isinstance(facts, dict) else {}
    fields = {
        k: k in facts and json.dumps(facts[k], sort_keys=True) == json.dumps(v, sort_keys=True)
        for k, v in spec["facts"].items()
    }
    claims = {
        "fact_keys": set(facts) == set(spec["facts"]),
        "disposition": state == spec["disposition"],
        "candidates": valid_ids and sorted(ids) == sorted(spec["candidates"]),
        "evidence": isinstance(paths, list)
        and all(isinstance(p, str) for p in paths)
        and set(spec["evidence"]) <= set(paths) <= set(scene_paths),
    }
    wrong = state == "identified" and (spec["disposition"] != "identified" or not claims["candidates"])
    return {
        "fact_pass": all(claims.values()) and all(fields.values()),
        "checks": claims,
        "fields": fields,
        "wrong_identification": wrong,
        "next_step_present": isinstance(answer.get("next_step"), str) and bool(answer["next_step"].strip()),
    }


def view(root, case, arm, work):
    shutil.copytree(root / "cases" / case / "scene", work / "scene")
    if arm in ("script", "skill") and (root / "package/scripts").is_dir():
        shutil.copytree(root / "package/scripts", work / "package/scripts")
    if arm == "skill":
        shutil.copytree(root / "package", work / "package", dirs_exist_ok=True)


def schedule(lock, config):
    plan = []
    arms = lock["arms"]
    rng = random.Random(config["seed"])
    for i in range(config["rounds"]):
        order = list(lock["cases"])
        rng.shuffle(order)
        for case in order:
            offset = (config["seed"] + lock["cases"].index(case) + i) % len(arms)
            for arm in arms[offset:] + arms[:offset]:
                plan.append({"case": case, "round": i + 1, "arm": arm})
    return plan


def run(root, lock, out, config):
    if lock != frozen(root):
        raise ValueError("Suite differs from lock")
    rounds, timeout = config["rounds"], config["timeout_seconds"]
    argv = config["command"]
    if (
        type(config.get("measure_evidence", False)) is not bool
        or type(rounds) is not int
        or rounds < 1
        or type(timeout) is not int
        or timeout < 1
        or not isinstance(argv, list)
        or not argv
        or not all(isinstance(x, str) for x in argv)
        or not config["model"].strip()
    ):
        raise ValueError("Invalid runner config")
    out.mkdir(parents=True, exist_ok=False)
    save(out / "lock.json", lock)
    save(out / "config.json", config)
    save(out / "engine.json", {"sha256": digest(Path(__file__)), "python": sys.version})
    plan = schedule(lock, config)
    save(out / "plan.json", plan)
    for index, item in enumerate(plan):
        case, arm = item["case"], item["arm"]
        trial = out / f"{index:04d}"
        trial.mkdir()
        with tempfile.TemporaryDirectory(prefix="skill-eval-") as tmp:
            work = Path(tmp)
            view(root, case, arm, work)
            case_dir = root / "cases" / case
            spec = load(case_dir / "rubric.json")
            prompt = (case_dir / "prompt.txt").read_text(encoding="utf-8") + "\n" + public_contract(case_dir, spec)
            if spec.get("profile") == "artifact":
                (work / "output").mkdir()
            save(trial / "input.json", {"prompt": prompt, "files": inventory(work)})
            timing = config.get("measure_evidence", False) and spec.get("profile", "diagnosis") == "diagnosis"
            journal = (trial / "findings.jsonl").resolve()
            timing_env = {k: v for k, v in os.environ.items() if k not in {"SKILL_EVAL_JOURNAL", "SKILL_EVAL_START_NS"}}
            if timing:
                journal.touch(exist_ok=False)
            start_ns = time.monotonic_ns()
            if timing:
                timing_env.update(SKILL_EVAL_JOURNAL=str(journal), SKILL_EVAL_START_NS=str(start_ns))
            answer = None
            error = None
            try:
                result = subprocess.run(
                    argv,
                    cwd=work,
                    input=prompt.encode(),
                    capture_output=True,
                    timeout=timeout,
                    env={**timing_env, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"},
                )
                (trial / "stdout").write_bytes(result.stdout)
                (trial / "stderr").write_bytes(result.stderr)
                if result.returncode:
                    raise ValueError(f"adapter exit {result.returncode}")
                answer = json.loads(result.stdout)
                if not isinstance(answer, dict):
                    raise ValueError("not a JSON object")
            except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
                error = type(exc).__name__ + ": " + str(exc)
                answer = None
                if isinstance(exc, subprocess.TimeoutExpired):
                    (trial / "stdout").write_bytes(exc.stdout or b"")
                    (trial / "stderr").write_bytes(exc.stderr or b"")
            elapsed = (time.monotonic_ns() - start_ns) / 1e9
            artifacts = {}
            if spec.get("profile") == "artifact" and error is None:
                for name in artifact_rules(spec):
                    src = work / "output" / name
                    parts = [
                        work / "output",
                        *(work / "output" / Path(*Path(name).parts[:i]) for i in range(1, len(Path(name).parts) + 1)),
                    ]
                    if any(
                        p.is_symlink() or (p.exists() and getattr(p.lstat(), "st_file_attributes", 0) & 1024)
                        for p in parts
                    ):
                        continue
                    if not src.is_file() or src.stat().st_nlink != 1 or src.stat().st_size > MAX_ARTIFACT_BYTES:
                        continue
                    dst = trial / "artifacts" / name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                    artifacts[name] = digest(dst)
            save(
                trial / "receipt.json",
                dict(
                    **item,
                    seconds=round(elapsed, 3),
                    error=error,
                    answer=answer,
                    artifacts=artifacts,
                    journal_sha256=digest(journal) if timing else None,
                ),
            )
    save(out / "results.json", replay(root, out))


def first_finding(path, sha, spec, scene_paths, seconds):
    if digest(path) != sha:
        raise ValueError("Evidence checkpoint journal changed since trial")
    first = None
    previous = -1
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        ns = entry["ns"]
        if type(ns) is not int or ns < 0 or ns < previous or ns / 1e9 > seconds + 0.001:
            raise ValueError("Invalid checkpoint timestamp")
        previous = ns
        if first is None and score(spec, entry["finding"], scene_paths)["fact_pass"]:
            first = ns / 1e9
    return first


def replay(root, out):
    lock = load(out / "lock.json")
    if lock != frozen(root):
        raise ValueError("Cannot replay changed suite")
    if load(out / "engine.json")["sha256"] != digest(Path(__file__)):
        raise ValueError("Replay with the recorded evaluator version")
    rows = []
    plan = load(out / "plan.json")
    if plan != schedule(lock, load(out / "config.json")):
        raise ValueError("Plan no longer matches frozen cases and recorded run config")
    for i, item in enumerate(plan):
        receipt_path = out / f"{i:04d}" / "receipt.json"
        r = (
            load(receipt_path)
            if receipt_path.exists()
            else {**item, "answer": None, "error": "missing trial", "seconds": None}
        )
        if any(item[k] != r[k] for k in item):
            raise ValueError("Receipt does not match predeclared trial")
        if receipt_path.exists() and r["error"] is None:
            raw = json.loads((receipt_path.parent / "stdout").read_bytes())
            if raw != r["answer"]:
                raise ValueError("Receipt answer differs from original adapter output")
        spec = load(root / "cases" / item["case"] / "rubric.json")
        if spec.get("profile") == "artifact":
            copies = receipt_path.parent / "artifacts"
            observed = inventory(copies)
            if observed != r.get("artifacts", {}):
                raise ValueError("Archived output changed since trial")
            checks = {}
            for name, expected in artifact_rules(spec).items():
                checks[name] = False
                if name in observed and r["error"] is None:
                    try:
                        value = load(copies / name)
                        checks[name] = json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True)
                    except (UnicodeError, ValueError):
                        pass
            result = {"profile": "artifact", "task_pass": all(checks.values()), "checks": checks}
        else:
            result = score(spec, r["answer"], inventory(root / "cases" / item["case"] / "scene"))
            result.update(profile="diagnosis", task_pass=result["fact_pass"])
            if load(out / "config.json").get("measure_evidence"):
                result["seconds_first_correct_evidence_submission"] = None
                if receipt_path.exists():
                    value = first_finding(
                        receipt_path.parent / "findings.jsonl",
                        r["journal_sha256"],
                        spec,
                        inventory(root / "cases" / item["case"] / "scene"),
                        r["seconds"],
                    )
                    if r["error"] is None:
                        result["seconds_first_correct_evidence_submission"] = value
        rows.append({**item, "kind": spec["kind"], "seconds": r["seconds"], "error": r["error"], **result})
    return rows


def review(root, out, packet):
    rows = replay(root, out)
    packet.mkdir(parents=True, exist_ok=False)

    key = {}
    questions = []
    for i, row in enumerate(rows):
        if row["profile"] != "diagnosis":
            continue
        case = root / "cases" / row["case"]
        trial = out / f"{i:04d}" / "receipt.json"
        spec = load(case / "rubric.json")
        token = secrets.token_hex(12)
        key[token] = i
        questions.append(
            {
                "id": token,
                "ticket": (case / "prompt.txt").read_text(encoding="utf-8"),
                "facts": spec["facts"],
                "disposition": spec["disposition"],
                "expected_candidates": spec["candidates"],
                "rubric": spec["next_step_rubric"],
                "answer": load(trial)["answer"] if trial.exists() else None,
            }
        )
    secrets.SystemRandom().shuffle(questions)
    if (out / "review-key.json").exists():
        raise ValueError("Review packet already issued")
    save(out / "review-key.json", key)
    save(packet / "questions.json", questions)


def ratings(root, out, path):
    rows = replay(root, out)
    key = load(out / "review-key.json")
    votes = load(path)
    if not isinstance(votes, dict) or set(votes) != set(key):
        raise ValueError("Need one rating per blinded answer")
    for token, i in key.items():
        vote = votes[token]
        if type(vote["score"]) is not int or vote["score"] not in (0, 1, 2) or not vote["reason"].strip():
            raise ValueError("Invalid rating")
        if vote["score"] and not rows[i]["next_step_present"]:
            raise ValueError("Missing next step cannot receive credit")
        rows[i]["next_step_score"] = vote["score"]
        rows[i]["next_step_reason"] = vote["reason"]
    return rows


def summarize(rows):
    """Separate data kinds and never allow one case's repeated trials to stand in for cases."""
    result = []
    keys = sorted({(r["kind"], r["case"], r["arm"]) for r in rows})
    for kind, case, arm in keys:
        group = [r for r in rows if (r["kind"], r["case"], r["arm"]) == (kind, case, arm)]
        times = [r["seconds"] for r in group if r["task_pass"] and r["seconds"] is not None]
        result.append(
            {
                "kind": kind,
                "case": case,
                "arm": arm,
                "attempts": len(group),
                "profile": group[0]["profile"],
                "task_passes": sum(r["task_pass"] for r in group),
                "fact_passes": sum(r["fact_pass"] for r in group) if group[0]["profile"] == "diagnosis" else None,
                "wrong_identifications": (
                    sum(r["wrong_identification"] for r in group) if group[0]["profile"] == "diagnosis" else None
                ),
                "operational_failures": sum(r["error"] is not None for r in group),
                "median_seconds_correct_final_answer": statistics.median(times) if times else None,
                "mean_next_step_score": (
                    statistics.mean(r["next_step_score"] for r in group)
                    if all("next_step_score" in r for r in group)
                    else None
                ),
            }
        )
        if any("seconds_first_correct_evidence_submission" in r for r in group):
            observations = [
                r["seconds_first_correct_evidence_submission"]
                for r in group
                if r.get("seconds_first_correct_evidence_submission") is not None
            ]
            result[-1]["trials_with_correct_evidence_submission"] = len(observations)
            result[-1]["median_seconds_first_correct_evidence_among_observed"] = (
                statistics.median(observations) if observations else None
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="cmd", required=True)
    for name, params in {
        "freeze": ("suite", "lock"),
        "verify": ("suite", "lock"),
        "run": ("suite", "lock", "config", "out"),
        "replay": ("suite", "out"),
        "review": ("suite", "out", "packet"),
        "ratings": ("suite", "out", "votes"),
        "summary": ("suite", "out"),
    }.items():
        p = subs.add_parser(name)
        for param in params:
            p.add_argument(param, type=Path)
        if name == "summary":
            p.add_argument("--votes", type=Path)
    a = parser.parse_args()
    root = a.suite.resolve()
    cmd = a.cmd
    for name in ("lock", "config", "out", "packet", "votes"):
        path = getattr(a, name, None)
        if path and (root == path.resolve() or root in path.resolve().parents):
            parser.error(f"{name} must be outside suite")
    if cmd == "freeze":
        if a.lock.exists():
            parser.error("Lock already exists")
        save(a.lock, frozen(root))
    elif cmd == "verify":
        if load(a.lock) != frozen(root):
            parser.error("Suite differs from lock")
    elif cmd == "run":
        run(root, load(a.lock), a.out, load(a.config))
    elif cmd == "review":
        review(root, a.out, a.packet)
    else:
        rows = ratings(root, a.out, a.votes) if getattr(a, "votes", None) else replay(root, a.out)
        print(json.dumps(summarize(rows) if cmd == "summary" else rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
