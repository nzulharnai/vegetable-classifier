"""Export the final model to ONNX, check it matches PyTorch, compare CPU speed.

pip install -r requirements-onnx.txt
python -m src.export_onnx   -> weights/model.onnx
"""
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from predict import load_model
from src.data import load_photo
from src.model import get_transforms
from src.train import ROOT

PATH = ROOT / "weights" / "model.onnx"


def median_ms(fn, xs, warmup=10):
    for x in xs[:warmup]:
        fn(x)
    times = []
    for x in xs:
        t = time.perf_counter()
        fn(x)
        times.append(1000 * (time.perf_counter() - t))
    return np.median(times), np.quantile(times, 0.95)


def main():
    net = load_model()
    torch.onnx.export(net, torch.zeros(1, 3, 224, 224), str(PATH), input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = torch.get_num_threads()
    sess = ort.InferenceSession(str(PATH), opts, providers=["CPUExecutionProvider"])
    df = pd.read_csv(ROOT / "data/splits.csv").sample(200, random_state=0)
    t = get_transforms(False)
    xs = [t(load_photo(ROOT / "data/Dataset" / p))[None] for p in df.path]

    def torch_run(x):
        with torch.inference_mode():
            return net(x).numpy()

    def onnx_run(x):
        return sess.run(None, {"image": x.numpy()})[0]

    diff = max(np.abs(torch_run(x) - onnx_run(x)).max() for x in xs)
    same = np.mean([torch_run(x).argmax() == onnx_run(x).argmax() for x in xs])
    print(f"200 photos: max logit difference {diff:.6f}, same class {same:.3f}")
    print(f"size {PATH.stat().st_size / 1e6:.1f} MB, threads {opts.intra_op_num_threads}")
    for name, fn in [("PyTorch", torch_run), ("ONNX Runtime", onnx_run)]:
        m, p = median_ms(fn, xs)
        print(f"{name}: network only median {m:.1f} ms, p95 {p:.1f} ms")


if __name__ == "__main__":
    main()
