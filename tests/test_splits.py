import pandas as pd

from src.data import CLASSES
from src.splits import make_splits


def _df():
    names = []
    for c in CLASSES:
        times = ["20251113_100000", "20260101_100000"] if c != "Raddish" else ["20251210_230636", "20251210_230730"]
        for t in times:
            names += [(c, f"{t}_frame_{i}.jpg") for i in range(6)]
    names += [("Carrot", "20251113_100030.jpg"), ("Carrot", "20260101_095900.jpg"), ("Raddish", "20251210_230725.jpg")]
    return pd.DataFrame([(f"{c}/{n}", c) for c, n in names], columns=["path", "label"])


def test_session_is_in_exactly_one_fold():
    s = make_splits(_df(), seed=42)
    assert (s.groupby("session").fold.nunique() == 1).all()


def test_every_class_is_in_both_folds():
    s = make_splits(_df(), seed=42)
    assert (s.groupby("label").fold.nunique() == 2).all()
    assert (s.groupby("label").random_split.nunique() == 2).all()


def test_still_goes_to_nearest_session():
    s = make_splits(_df(), seed=42).set_index("path")
    assert s.loc["Carrot/20251113_100030.jpg", "session"] == "Carrot/20251113_100000"
    assert s.loc["Carrot/20260101_095900.jpg", "session"] == "Carrot/20260101_100000"
    assert s.loc["Raddish/20251210_230725.jpg", "session"] == "Raddish/20251210_230730"


def test_january_is_fold_1_and_radish_is_split_by_video():
    s = make_splits(_df(), seed=42).set_index("session").fold
    assert set(s.loc["Squash/20260101_100000"]) == {1}
    assert set(s.loc["Squash/20251113_100000"]) == {0}
    assert set(s.loc["Raddish/20251210_230636"]) == {0}
    assert set(s.loc["Raddish/20251210_230730"]) == {1}


def test_reproducible():
    pd.testing.assert_frame_equal(make_splits(_df(), seed=42), make_splits(_df(), seed=42))
