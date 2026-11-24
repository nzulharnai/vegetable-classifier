import pandas as pd
import pytest
import torch
from PIL import Image

from src.model import build_model, get_transforms
from src.train import select


def test_output_shape():
    m = build_model(6, pretrained=False).eval()
    with torch.inference_mode():
        assert m(torch.zeros(2, 3, 224, 224)).shape == (2, 6)


@pytest.mark.parametrize("train", [True, False])
def test_transform_shape_for_any_aspect(train):
    assert get_transforms(train)(Image.new("RGB", (2084, 900))).shape == (3, 224, 224)


def _splits():
    return pd.DataFrame({
        "path": list("abcdef"),
        "fold": [0, 0, 0, 1, 1, 1],
        "random_split": [0, 1, 0, 1, 0, 1],
        "author_split": ["train", "test|train", "test", "train|val", "val", ""],
    })


def test_select_group_never_mixes_folds():
    tr, te = select(_splits(), "group", 1)
    assert set(tr.fold) == {0} and set(te.fold) == {1}


def test_select_author_keeps_overlap_on_purpose():
    tr, te = select(_splits(), "author", 0)
    assert list(tr.path) == ["a", "b", "d"] and list(te.path) == ["b", "c"]


def test_select_full_has_no_test():
    tr, te = select(_splits(), "full", 0)
    assert len(tr) == 6 and len(te) == 0
