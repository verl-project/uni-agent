"""Compare replayed trials by incident, without treating rounds as independent incidents."""

import argparse
import json
import math
import sys
from collections import defaultdict
from fractions import Fraction


def compare(rows, control, metric):
    grouped = defaultdict(dict)
    for row in rows:
        if row["arm"] not in ("skill", control):
            continue
        key = row["kind"], row["case"], row["round"]
        if row["arm"] in grouped[key]:
            raise ValueError("duplicate arm in one case and round")
        if metric not in row or row[metric] is None:
            raise ValueError(f"missing {metric}; supply completed ratings for next-step comparisons")
        grouped[key][row["arm"]] = int(row[metric])
    cases = defaultdict(list)
    for (kind, case, _), pair in grouped.items():
        if set(pair) != {"skill", control}:
            raise ValueError("unpaired trial; use full replay output including operational failures")
        cases[kind, case].append(pair["skill"] - pair[control])
    result = {}
    for kind in sorted({k for k, _ in cases}):
        deltas = {case: Fraction(sum(x), len(x)) for (k, case), x in cases.items() if k == kind}
        wins = sum(d > 0 for d in deltas.values())
        losses = sum(d < 0 for d in deltas.values())
        n = wins + losses
        # Two-sided exact sign test. Ties carry no directional information. Different
        # cases must represent independent incidents for the p-value to be meaningful.
        p = min(1, 2 * Fraction(sum(math.comb(n, i) for i in range(min(wins, losses) + 1)), 2**n)) if n else None
        result[kind] = {
            "comparator": control,
            "metric": metric,
            "units": "skill minus control; lower is better only for wrong_identification",
            "case_mean_delta": {k: float(v) for k, v in deltas.items()},
            "positive": wins,
            "negative": losses,
            "ties": len(deltas) - n,
            "mean_delta_equal_case_weight": float(sum(deltas.values()) / len(deltas)),
            "two_sided_sign_p_if_independent": float(p) if p is not None else None,
        }
    if not result:
        raise ValueError("no paired trials")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", choices=["baseline", "script"], required=True)
    p.add_argument(
        "--metric", choices=["task_pass", "fact_pass", "wrong_identification", "next_step_score"], default="task_pass"
    )
    args = p.parse_args()
    print(json.dumps(compare(json.load(sys.stdin), args.control, args.metric), indent=2))
