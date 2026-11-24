"""CPU time per photo: python -m src.benchmark"""
import time

import numpy as np
import pandas as pd
import torch

from predict import load_model
from src.data import load_photo
from src.model import get_transforms
from src.train import ROOT

df = pd.read_csv(ROOT / "data/splits.csv")
still = ~df.path.str.contains("_frame_")
t = time.perf_counter()
net = load_model()
print(f"load model {1000 * (time.perf_counter() - t):.0f} ms, torch threads {torch.get_num_threads()}")
tr = get_transforms(False)
for name, part in [("frames 200x200", df[~still].sample(200, random_state=0)), ("photos 10-12 MP", df[still])]:
    times, infer = [], []
    with torch.inference_mode():
        for p in part.path:
            t0 = time.perf_counter()
            x = tr(load_photo(ROOT / "data/Dataset" / p))[None]
            t1 = time.perf_counter()
            net(x).softmax(1)
            t2 = time.perf_counter()
            times.append(1000 * (t2 - t0))
            infer.append(1000 * (t2 - t1))
    print(f"{name}: n={len(part)} total median {np.median(times):.1f} ms, p95 {np.quantile(times, .95):.1f} ms; network only median {np.median(infer):.1f} ms")
