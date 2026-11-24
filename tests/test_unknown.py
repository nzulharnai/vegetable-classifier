import pandas as pd
import pytest

pytest.importorskip("sklearn")
from src.unknown import choose, measure


def _d():
    known = pd.DataFrame({"label": ["Carrot"] * 10, "pred": ["Carrot"] * 9 + ["Squash"],
                          "confidence": [0.1 * k for k in range(1, 11)], "unknown": False, "excluded": "Potato"})
    unk = pd.DataFrame({"label": ["Potato"] * 4, "pred": ["Carrot"] * 4,
                        "confidence": [0.05, 0.1, 0.5, 0.99], "unknown": True, "excluded": "Potato"})
    return pd.concat([known, unk], ignore_index=True)


def test_choose_keeps_the_requested_share_of_known():
    t = choose(_d(), keep=0.9)
    assert (_d().query("not unknown").confidence >= t).mean() == pytest.approx(0.9, abs=0.051)


def test_measure_counts_rejected_unknown():
    r = measure(_d(), 0.3)
    assert r["unknown_rejected"] == 0.5
    assert r["known_kept"] == 0.8
    assert r["rejected_by_excluded_class"] == {"Potato": 0.5}
