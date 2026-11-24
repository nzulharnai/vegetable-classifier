"""Two folds by shooting session, plus a random 2-fold split for comparison.

Run: python -m src.splits  -> data/splits.csv
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data import CLASSES, parse_name

ROOT = Path(__file__).resolve().parents[1]
JANUARY = pd.Timestamp("2026-01-01")


def make_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """df has columns path ("Label/name.jpg") and label."""
    out = df.sort_values("path").reset_index(drop=True).copy()
    parsed = [parse_name(p.split("/")[-1]) for p in out.path]
    times = [t for t, _ in parsed]
    is_video = [f is not None for _, f in parsed]
    videos = {c: sorted({t for t, v, l in zip(times, is_video, out.label) if v and l == c}) for c in CLASSES}

    # A still joins the video of its class that started closest in time.
    start = [t if v else min(videos[l], key=lambda s: abs(s - t))
             for t, v, l in zip(times, is_video, out.label)]
    out["session"] = [f"{l}/{s:%Y%m%d_%H%M%S}" for l, s in zip(out.label, start)]

    # January against the earlier dates; a class shot on one day is split by video order.
    fold = []
    for label, s in zip(out.label, start):
        vs = videos[label]
        if len({v.date() for v in vs}) > 1:
            fold.append(int(s >= JANUARY))
        else:
            fold.append(int(vs.index(s) >= len(vs) / 2))
    out["fold"] = fold

    out["random_split"] = 0
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    out.loc[list(skf.split(out, out.label))[1][1], "random_split"] = 1
    return out


def main() -> None:
    files = sorted((ROOT / "data" / "Dataset").rglob("*.jpg"))
    df = pd.DataFrame({"path": [f"{p.parent.name}/{p.name}" for p in files],
                       "label": [p.parent.name for p in files]})
    out = make_splits(df, seed=42)
    a = pd.read_csv(ROOT / "data" / "author_split.csv")
    a["path"] = a["label"] + "/" + a["name"]
    author = a.groupby("path").author_split.agg(lambda s: "|".join(sorted(s)))
    out["author_split"] = out.path.map(author).fillna("")
    out.to_csv(ROOT / "data" / "splits.csv", index=False)
    print(out.groupby(["label", "fold"]).size().unstack())
    print(out.groupby(["fold", "session"]).size())
    print(out.groupby(["label", "random_split"]).size().unstack())
    print(out.author_split.value_counts())


if __name__ == "__main__":
    main()
