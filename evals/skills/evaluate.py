"""Run a frozen three-arm skill evaluation. Python 3.10+, standard library only."""

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
from pathlib import Path

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


def cases(root):
    answer = []
    for case in sorted((root / "cases").iterdir()):
        if not case.is_dir():
            raise ValueError(f"Unexpected file among cases: {case}")
        spec = load(case / "rubric.json")
        if spec["kind"] not in KINDS or spec["disposition"] not in STATES:
            raise ValueError(f"Invalid provenance kind or disposition: {case.name}")
        ids = spec["candidates"]
        if not isinstance(ids, list) or not all(isinstance(x, str) and x for x in ids) or len(ids) != len(set(ids)):
            raise ValueError("candidates must be unique nonempty strings")
        if (spec["disposition"] == "identified" and len(ids) != 1) or (
            spec["disposition"] == "ambiguous" and len(ids) < 2
        ):
            raise ValueError("Disposition incompatible with expected candidates")
        if (
            not spec["provenance"].strip()
            or not spec["next_step_rubric"].strip()
            or not isinstance(spec["facts"], dict)
        ):
            raise ValueError(f"Missing provenance, grading rubric, or facts: {case.name}")
        paths = set(inventory(case / "scene"))
        if (
            not paths
            or not spec["evidence"]
            or not set(spec["evidence"]) <= paths
            or not (case / "prompt.txt").read_text(encoding="utf-8").strip()
        ):
            raise ValueError(f"Missing input or evidence path: {case.name}")
        answer.append(case.name)
    if (
        not answer
        or not (root / "package/SKILL.md").is_file()
        or not inventory(root / "package/scripts")
        or not (root / "protocol.md").read_text(encoding="utf-8").strip()
    ):
        raise ValueError("Suite requires cases, a protocol, SKILL.md and shared scripts")
    return answer


def frozen(root):
    files = inventory(root)
    return {"version": 1, "cases": cases(root), "sha256": files}


def score(spec, answer):
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
        "disposition": state == spec["disposition"],
        "candidates": valid_ids and sorted(ids) == sorted(spec["candidates"]),
        "evidence": isinstance(paths, list)
        and all(isinstance(p, str) for p in paths)
        and set(spec["evidence"]) <= set(paths),
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
    if arm in ("script", "skill"):
        shutil.copytree(root / "package/scripts", work / "package/scripts")
    if arm == "skill":
        shutil.copytree(root / "package", work / "package", dirs_exist_ok=True)


def schedule(lock, config):
    plan = []
    rng = random.Random(config["seed"])
    for i in range(config["rounds"]):
        order = list(lock["cases"])
        rng.shuffle(order)
        for case in order:
            offset = (config["seed"] + lock["cases"].index(case) + i) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                plan.append({"case": case, "round": i + 1, "arm": arm})
    return plan


def run(root, lock, out, config):
    if lock != frozen(root):
        raise ValueError("Suite differs from lock")
    rounds, timeout = config["rounds"], config["timeout_seconds"]
    argv = config["command"]
    if (
        type(rounds) is not int
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
            prompt = (root / "cases" / case / "prompt.txt").read_text(encoding="utf-8") + "\n" + RESPONSE
            save(trial / "input.json", {"prompt": prompt, "files": inventory(work)})
            start = time.monotonic()
            answer = None
            error = None
            try:
                result = subprocess.run(
                    argv,
                    cwd=work,
                    input=prompt.encode(),
                    capture_output=True,
                    timeout=timeout,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"},
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
            save(
                trial / "receipt.json",
                dict(**item, seconds=round(time.monotonic() - start, 3), error=error, answer=answer),
            )
    save(out / "results.json", replay(root, out))


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
        result = score(spec, r["answer"])
        rows.append({**item, "kind": spec["kind"], "seconds": r["seconds"], "error": r["error"], **result})
    return rows


def review(root, out, packet):
    rows = replay(root, out)
    packet.mkdir(parents=True, exist_ok=False)

    key = {}
    questions = []
    for i, row in enumerate(rows):
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
        times = [r["seconds"] for r in group if r["fact_pass"] and r["seconds"] is not None]
        result.append(
            {
                "kind": kind,
                "case": case,
                "arm": arm,
                "attempts": len(group),
                "fact_passes": sum(r["fact_pass"] for r in group),
                "wrong_identifications": sum(r["wrong_identification"] for r in group),
                "operational_failures": sum(r["error"] is not None for r in group),
                "median_seconds_correct_final_answer": statistics.median(times) if times else None,
                "mean_next_step_score": (
                    statistics.mean(r["next_step_score"] for r in group)
                    if all("next_step_score" in r for r in group)
                    else None
                ),
            }
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
