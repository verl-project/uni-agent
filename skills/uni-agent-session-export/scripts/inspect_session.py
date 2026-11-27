"""Read-only Gateway export index; paths in the report are relative to the archive."""

import argparse
import json
import os
import stat
from collections import Counter
from pathlib import Path

META, SNAP, ROWS = "session_metadata.json", "debug_snapshot.json", "trajectories.jsonl"
NAMES = {META, SNAP, ROWS}
MAX_EXAMPLES = 8


def blocked(path):
    try:
        info = path.lstat()
    except OSError:
        return False  # The reader reports missing and unreadable files separately.
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def reader(path, mode, **kwargs):
    if not stat.S_ISREG(path.stat().st_mode):
        raise OSError("not a regular file")
    return path.open(mode, **kwargs)


def obj(p):
    if blocked(p):
        return {}, "link_skipped"
    try:
        with reader(p, "r", encoding="utf-8-sig") as f:
            value = json.load(f)
        return (value, "present") if isinstance(value, dict) else ({}, "invalid_object")
    except FileNotFoundError:
        return {}, "missing"
    except (UnicodeError, ValueError):
        return {}, "invalid_json"
    except OSError:
        return {}, "unreadable"


def session_id(obj):
    v = obj.get("session_id")
    return v if isinstance(v, str) and v.strip() else None


def fields(x, needle, prefix="$"):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from fields(v, needle, prefix + "." + k)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from fields(v, needle, f"{prefix}[{i}]")
    elif isinstance(x, str) and needle in x:
        yield prefix


def inspect(root, needle=None):
    out = {"exports": [], "matches": [], "walk_errors": []}
    matches = out["matches"]

    def relative(path):
        return path.relative_to(root).as_posix()

    def record(doc, path, row=None):
        if not needle:
            return
        found = []
        total = 0
        for location in fields(doc, needle):
            total += 1
            if len(found) < MAX_EXAMPLES:
                found.append(location)
        if found:
            hit = {
                "file": relative(path),
                "session_id": session_id(doc),
                "fields": found,
                "field_count": total,
            }
            if row is not None:
                hit["line"] = row
            matches.append(hit)

    def onerror(exc):
        out["walk_errors"].append({"directory": relative(Path(exc.filename)), "error": type(exc).__name__})

    for folder, dirs, files in os.walk(root, followlinks=False, onerror=onerror):
        folder = Path(folder)
        dirs[:] = sorted(d for d in dirs if d != ".git" and not blocked(folder / d))
        if needle:
            for name in sorted(f for f in files if f.lower().endswith(".json") and f not in NAMES):
                p = folder / name
                doc, state = obj(p)
                if state == "present":
                    record(doc, p)
        if not NAMES.intersection(files):
            continue
        meta, ms = obj(folder / META)
        snap, ss = obj(folder / SNAP)
        record(meta, folder / META)
        record(snap, folder / SNAP)
        value = meta.get("trajectories_count")
        declared = value if type(value) is int and value >= 0 else None
        mid, sid = session_id(meta), session_id(snap)
        if ms == "present" and declared is None:
            ms = "count_missing_or_invalid"
        directory = relative(folder)
        counts = Counter()
        invalid = []
        bad = good = unknown = 0
        state = "missing"
        n = None
        path = folder / ROWS
        if blocked(path):
            state = "link_skipped"
        else:
            try:
                with reader(path, "rb") as stream:
                    for line_no, raw in enumerate(stream, 1):
                        if not raw.strip():
                            continue
                        try:
                            row = json.loads(raw.decode("utf-8-sig" if line_no == 1 else "utf-8"))
                            if not isinstance(row, dict):
                                raise ValueError("not an object")
                        except (UnicodeError, ValueError):
                            bad += 1
                            if len(invalid) < MAX_EXAMPLES:
                                invalid.append(line_no)
                            continue
                        good += 1
                        rid = session_id(row)
                        if rid is not None:
                            counts[rid] += 1
                        else:
                            unknown += 1
                        record(row, path, line_no)
                n = good
                state = "invalid_rows" if bad else ("nonempty" if good else "empty")
            except FileNotFoundError:
                pass
            except OSError:
                state = "unreadable"
        ids = set(counts) | {s for s in (mid, sid) if s is not None}
        snapshot_state = snap.get("session_state")
        snapshot_count = snapshot_state.get("num_trajectories") if isinstance(snapshot_state, dict) else None
        if type(snapshot_count) is not int or snapshot_count < 0:
            snapshot_count = None
        out["exports"].append(
            {
                "directory": directory,
                "metadata": {"state": ms, "id": mid, "declared": declared},
                "snapshot": {"state": ss, "id": sid, "snapshot_count": snapshot_count},
                "jsonl": {
                    "state": state,
                    "valid": n,
                    "by_id": dict(sorted(counts.items())),
                    "without_id": unknown if n is not None else None,
                    "invalid_count": bad if n is not None else None,
                    "invalid_line_examples": invalid,
                },
                "id_conflict": len(ids) > 1,
                "comparison": (
                    "unknown" if declared is None or n is None or bad else ("equal" if declared == n else "different")
                ),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--match",
        metavar="LITERAL",
        help="case-sensitive substring in JSON strings; returns locations, not message content",
    )
    a = parser.parse_args()
    root = a.directory.resolve()
    if not root.is_dir():
        parser.error("expected archive directory")
    if a.match == "":
        parser.error("match must not be empty")
    print(json.dumps(inspect(root, a.match), ensure_ascii=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
