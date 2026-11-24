"""Metrics, confusion matrices and Grad-CAM from runs/*.csv.

Run: python -m src.evaluate
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from src.data import CLASSES
from src.model import SIZE, build_model, get_transforms
from src.train import ROOT

OUT = ROOT / "reports"
RUNS = ["group_0", "group_1", "random_0", "random_1", "author_0"]


def summarize(d: pd.DataFrame) -> dict:
    p, r, f, n = precision_recall_fscore_support(d.label, d.pred, labels=CLASSES, zero_division=0)
    return {
        "n": len(d),
        "accuracy": round(float((d.label == d.pred).mean()), 4),
        "macro_f1": round(float(f1_score(d.label, d.pred, labels=CLASSES, average="macro", zero_division=0)), 4),
        "per_class": {c: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                          "f1": round(float(f[i]), 3), "n": int(n[i])} for i, c in enumerate(CLASSES)},
        "confusion": confusion_matrix(d.label, d.pred, labels=CLASSES).tolist(),
    }


def gradcam(net, x, cls):
    """Grad-CAM on the last convolutional layer, scaled to [0, 1]."""
    keep = {}

    def hook(_, __, out):
        out.retain_grad()
        keep["a"] = out

    h = net.features[-1].register_forward_hook(hook)
    try:
        net.zero_grad()
        net(x)[0, cls].backward()
    finally:
        h.remove()
    a, g = keep["a"], keep["a"].grad
    cam = torch.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, (SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0].detach()
    return (cam / cam.max().clamp(min=1e-8)).numpy()


def pick_examples(d):
    """Six most confident errors (one per session and confusion type), four correct."""
    bad = d[d.label != d.pred].sort_values("confidence", ascending=False)
    bad = bad.drop_duplicates(["run", "session", "label", "pred"]).head(6)
    good = d[d.label == d.pred].groupby("label").sample(1, random_state=42)
    return pd.concat([bad, good.sample(4, random_state=42)])


def main():
    sessions = pd.read_csv(ROOT / "data/splits.csv").set_index("path").session
    res, frames = {}, {}
    for r in RUNS:
        frames[r] = pd.read_csv(ROOT / "runs" / f"{r}.csv").assign(run=r)
        frames[r]["session"] = frames[r].path.map(sessions)
        res[r] = summarize(frames[r])
    for s in ("group", "random"):
        res[f"{s}_both"] = summarize(pd.concat([frames[f"{s}_0"], frames[f"{s}_1"]]))
    g = pd.concat([frames["group_0"], frames["group_1"]], ignore_index=True)
    per_session = g.assign(ok=g.label == g.pred).groupby(["run", "session"]).ok.agg(["size", "mean"]).round(3)
    per_session.to_csv(OUT / "per_session.csv")
    g["still"] = ~g.path.str.contains("_frame_")
    res["group_stills_only"] = summarize(g[g.still])
    res["group_frames_only"] = summarize(g[~g.still])
    wrong = g.label != g.pred
    res["group_confidence"] = {"wrong_n": int(wrong.sum()),
                               "wrong_median": round(float(g[wrong].confidence.median()), 3),
                               "wrong_above_0.9": round(float((g[wrong].confidence > 0.9).mean()), 3),
                               "right_median": round(float(g[~wrong].confidence.median()), 3)}
    (OUT / "metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    for ax, r in zip(axes, ["group_0", "group_1", "random_0"]):
        m = np.array(res[r]["confusion"])
        ax.imshow(m / m.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                if m[i, j]:
                    ax.text(j, i, m[i, j], ha="center", va="center", color="white" if i == j else "black")
        ax.set_xticks(range(6), CLASSES, rotation=45)
        ax.set_yticks(range(6), CLASSES)
        ax.set(title=f"{r}: acc {res[r]['accuracy']:.3f}, macro-F1 {res[r]['macro_f1']:.3f}",
               xlabel="predicted", ylabel="true")
    fig.tight_layout()
    fig.savefig(OUT / "confusion.png", dpi=90)

    t = get_transforms(False)
    nets = {}
    for r in ("group_0", "group_1"):
        nets[r] = build_model(pretrained=False).eval()
        nets[r].load_state_dict(torch.load(ROOT / "weights" / f"{r}.pt", weights_only=True))
    ex = pick_examples(g)
    ex.to_csv(OUT / "gradcam_examples.csv", index=False)
    fig, axes = plt.subplots(4, 5, figsize=(15, 12.5))
    for k, (_, e) in enumerate(ex.iterrows()):
        with Image.open(ROOT / "data/Dataset" / e.path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        cam = gradcam(nets[e.run], t(im)[None], CLASSES.index(e.pred))
        row, col = 2 * (k // 5), k % 5
        axes[row, col].imshow(im.resize((SIZE, SIZE)))
        axes[row, col].set_title(f"{'OK' if e.label == e.pred else 'ERROR'} true {e.label}\npred {e.pred} {e.confidence:.2f}", fontsize=10)
        axes[row + 1, col].imshow(im.resize((SIZE, SIZE)))
        axes[row + 1, col].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "gradcam.png", dpi=70)
    for k in ("group_0", "group_1", "group_both", "random_both", "author_0", "group_stills_only", "group_frames_only"):
        print(k, res[k]["n"], res[k]["accuracy"], res[k]["macro_f1"], {c: v["recall"] for c, v in res[k]["per_class"].items()})
    print(res["group_confidence"])
    print(per_session.to_string())
    for r in ("group_0", "group_1"):
        print(r, pd.crosstab(frames[r].label, frames[r].pred).to_string(), sep="\n")


if __name__ == "__main__":
    main()
