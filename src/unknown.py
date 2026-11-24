"""Confidence threshold for `unknown`, from leave-one-class-out runs.

Each run trains without one class; photos of that class play an unknown vegetable.
Rule fixed in advance: the threshold keeps 90% of photos of known classes. It is
chosen on the runs tested on one fold and checked on the runs tested on the other.

Run: python -m src.unknown
"""
import json

import numpy as np
import pandas as pd

from src.data import CLASSES
from src.train import ROOT

KEEP = 0.90


def load(fold):
    parts = []
    for c in CLASSES:
        d = pd.read_csv(ROOT / "runs" / f"group_{fold}_no{c}.csv")
        parts.append(d.assign(excluded=c, unknown=d.label == c))
    return pd.concat(parts, ignore_index=True)


def choose(d, keep=KEEP):
    """Largest threshold that still keeps the given share of known photos."""
    return float(np.quantile(d[~d.unknown].confidence, 1 - keep))


def measure(d, t):
    known, unk = d[~d.unknown], d[d.unknown]
    accepted = known[known.confidence >= t]
    return {
        "threshold": round(t, 4),
        "known_kept": round(float((known.confidence >= t).mean()), 3),
        "unknown_rejected": round(float((unk.confidence < t).mean()), 3),
        "accuracy_of_accepted_known": round(float((accepted.label == accepted.pred).mean()), 3),
        "accuracy_of_all_known": round(float((known.label == known.pred).mean()), 3),
        "rejected_by_excluded_class": {c: round(float((g.confidence < t).mean()), 3)
                                       for c, g in unk.groupby("excluded")},
    }


def main():
    runs = {f: load(f) for f in (0, 1)}
    t = {f: choose(runs[f]) for f in runs}
    res = {"chosen_on_fold_1_checked_on_fold_0": measure(runs[0], t[1]),
           "chosen_on_fold_0_checked_on_fold_1": measure(runs[1], t[0])}
    res["threshold_both_folds"] = round(choose(pd.concat(runs.values())), 4)
    (ROOT / "reports" / "unknown.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
