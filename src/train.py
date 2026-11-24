"""Train one split.

python -m src.train --split group --fold 0 [--data DIR] [--limit N]

--fold F is the fold used for testing (group and random). Epoch counts are fixed
in advance: nothing is chosen by looking at the test part.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data import CLASSES, load_photo
from src.model import build_model, get_transforms

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def select(df: pd.DataFrame, split: str, fold: int):
    if split == "group":
        return df[df.fold != fold], df[df.fold == fold]
    if split == "random":
        return df[df.random_split != fold], df[df.random_split == fold]
    if split == "author":
        parts = df.author_split.fillna("").str.split("|")
        return df[parts.map(lambda p: "train" in p)], df[parts.map(lambda p: "test" in p)]
    if split == "full":
        return df, df.iloc[0:0]
    raise ValueError(split)


def drop_class(tr: pd.DataFrame, te: pd.DataFrame, label: str):
    """Leave one class out of training; in the test part it plays an unknown vegetable."""
    return tr[tr.label != label], te


class Photos(Dataset):
    def __init__(self, df, data, transform):
        self.paths, self.labels, self.data, self.t = list(df.path), list(df.label), Path(data), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = load_photo(self.data / self.paths[i])
        return self.t(im), CLASSES.index(self.labels[i])


def run_epoch(net, loader, opt, loss_fn, dev, frozen):
    net.train()
    if frozen:
        net.features.eval()  # keep ImageNet batch-norm statistics while only the head learns
    total = n = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        opt.step()
        total, n = total + loss.item() * len(y), n + len(y)
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["group", "random", "author", "full"])
    ap.add_argument("--fold", type=int, default=0, choices=[0, 1])
    ap.add_argument("--data", default=str(ROOT / "data" / "Dataset"))
    ap.add_argument("--exclude", choices=CLASSES, help="class left out of training (unknown experiment)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: photos per class")
    ap.add_argument("--epochs-head", type=int, default=3)
    ap.add_argument("--epochs-full", type=int, default=7)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT))
    a = ap.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(ROOT / "data" / "splits.csv")
    tr, te = select(df, a.split, a.fold)
    if a.exclude:
        tr, te = drop_class(tr, te, a.exclude)
    if a.limit:
        tr, te = tr.groupby("label").head(a.limit), te.groupby("label").head(a.limit)
    print(f"{a.split} fold {a.fold}: train {len(tr)}, test {len(te)}, device {dev}", flush=True)

    counts = tr.label.value_counts().reindex(CLASSES)
    w = (len(tr) / (counts.notna().sum() * counts)).fillna(0.0)
    weight = torch.tensor(w.values, dtype=torch.float32, device=dev)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    loader = DataLoader(Photos(tr, a.data, get_transforms(True)), 64, shuffle=True, drop_last=len(tr) > 64,
                        num_workers=a.workers, generator=torch.Generator().manual_seed(SEED))
    net = build_model(len(CLASSES)).to(dev)
    for phase, epochs, lr in [("head", a.epochs_head, 1e-3), ("full", a.epochs_full, 3e-4)]:
        for p in net.features.parameters():
            p.requires_grad = phase == "full"
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(epochs, 1))
        for e in range(epochs):
            loss = run_epoch(net, loader, opt, loss_fn, dev, phase == "head")
            print(phase, e + 1, f"loss {loss:.4f}", flush=True)
            sched.step()

    out, name = Path(a.out), f"{a.split}_{a.fold}" + (f"_no{a.exclude}" if a.exclude else "")
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "weights").mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / "weights" / ("model.pt" if a.split == "full" else f"{name}.pt"))
    if len(te):
        net.eval()
        probs = []
        with torch.inference_mode():
            for x, _ in DataLoader(Photos(te, a.data, get_transforms(False)), 64, num_workers=a.workers):
                probs.append(net(x.to(dev)).softmax(1).cpu().numpy())
        probs = np.concatenate(probs)
        res = te[["path", "label"]].assign(pred=[CLASSES[i] for i in probs.argmax(1)], confidence=probs.max(1))
        for i, c in enumerate(CLASSES):
            res[f"p_{c}"] = probs[:, i]
        res.to_csv(out / "runs" / f"{name}.csv", index=False)
        print("accuracy", round((res.pred == res.label).mean(), 4))


if __name__ == "__main__":
    main()
