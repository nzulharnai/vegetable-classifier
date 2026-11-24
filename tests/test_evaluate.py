import pandas as pd
import torch

from src.evaluate import gradcam, summarize
from src.model import build_model


def test_summarize():
    d = pd.DataFrame({"label": ["Carrot", "Carrot", "Squash", "Potato"],
                      "pred": ["Carrot", "Squash", "Squash", "Potato"]})
    s = summarize(d)
    assert s["n"] == 4 and s["accuracy"] == 0.75
    assert s["per_class"]["Carrot"]["recall"] == 0.5
    assert s["per_class"]["Squash"]["precision"] == 0.5
    assert sum(map(sum, s["confusion"])) == 4


def test_gradcam_is_a_normalised_map():
    cam = gradcam(build_model(pretrained=False).eval(), torch.randn(1, 3, 224, 224), 0)
    assert cam.shape == (224, 224)
    assert cam.min() >= 0 and cam.max() <= 1
