"""Similarity statistics and contact sheets from the scan made by src.eda."""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from src.eda import OUT, ROOT

NEAR_PHASH = 8  # of 64 bits


def load():
    df = pd.read_csv(OUT / "images.csv", parse_dates=["time"])
    f = np.load(OUT / "features.npz")
    emb = f["emb"] / np.linalg.norm(f["emb"], axis=1, keepdims=True)
    df["video"] = np.where(df.frame.notna(), df.label + "/" + df.time.dt.strftime("%Y%m%d_%H%M%S"), "")
    df["unit"] = np.where(df.video != "", df.video, df.label + "/still_" + df.time.dt.strftime("%Y%m%d"))
    return df, f["phash"], emb


def sheet(df, name, n=6, size=170):
    units = sorted(df.unit.unique())
    img = Image.new("RGB", (n * size, len(units) * (size + 16)), "white")
    d = ImageDraw.Draw(img)
    for r, u in enumerate(units):
        g = df[df.unit == u].sort_values(["time", "frame"])
        d.text((4, r * (size + 16)), f"{u}  n={len(g)}", fill="black")
        for c, k in enumerate(np.unique(np.linspace(0, len(g) - 1, n).astype(int))):
            im = ImageOps.exif_transpose(Image.open(ROOT / "data/Dataset" / g.path.iloc[k])).convert("RGB")
            img.paste(ImageOps.pad(im, (size - 4, size - 4)), (c * size, r * (size + 16) + 14))
    img.save(OUT / name, quality=85)


def main():
    df, ph, emb = load()
    print("errors", df.error.notna().sum(), "md5 dup", df.md5.duplicated().sum())
    print(df.groupby(["width", "height"]).size().to_string())
    print("exif", df.orientation.value_counts().to_dict())
    print(df.groupby("unit")[["brightness", "sharpness"]].mean().round(0).assign(n=df.groupby("unit").size()).to_string())
    bits = ph.astype(np.float32) * 2 - 1
    ham = (64 - bits @ bits.T) / 2
    cos = emb @ emb.T
    np.fill_diagonal(ham, 99)
    np.fill_diagonal(cos, -1)
    lab, unit = df.label.to_numpy(object), df.unit.to_numpy(object)
    same_unit = unit[:, None] == unit[None]
    same_lab = lab[:, None] == lab[None]
    nxt = []
    for _, g in df[df.video != ""].groupby("video"):
        i = g.sort_values("frame").index.values
        fr = df.frame.values[i]
        nxt += [(a, b) for a, b, k in zip(i, i[1:], np.diff(fr)) if k == 1]
    a, b = np.array(nxt).T
    print("adjacent frames", len(a), "phash median", np.median(ham[a, b]), "share<=8", (ham[a, b] <= NEAR_PHASH).mean().round(3), "cos median", np.median(cos[a, b]).round(3))
    for nm, m in [("same unit", same_unit), ("other unit same class", ~same_unit & same_lab), ("other class", ~same_lab)]:
        h = np.where(m, ham, 99).min(1)
        c = np.where(m, cos, -1).max(1)
        print(f"nearest in {nm}: phash<=8 {(h <= NEAR_PHASH).mean():.3f}  cos median {np.median(c):.3f} p95 {np.quantile(c, .95):.3f}")
    us = sorted(set(unit))
    rows = []
    for i, u in enumerate(us):
        for v in us[i + 1:]:
            s = np.ix_(unit == u, unit == v)
            rows.append((u, v, cos[s].mean(), cos[s].max(), int((ham[s] <= NEAR_PHASH).sum())))
    t = pd.DataFrame(rows, columns=["a", "b", "cos_mean", "cos_max", "phash_pairs"])
    t.to_csv(OUT / "near_dups.csv", index=False)
    same = t[t.a.str.split("/").str[0] == t.b.str.split("/").str[0]]
    print(same.sort_values("cos_mean", ascending=False).round(3).head(30).to_string())
    diff = t.drop(same.index)
    print(diff.sort_values("cos_mean", ascending=False).round(3).head(10).to_string())
    nn = cos.argmax(1)
    print("1-NN acc, all images:", (lab[nn] == lab).mean().round(3))
    nn2 = np.where(same_unit, -1, cos).argmax(1)
    ok = lab[nn2] == lab
    print("1-NN acc, other units only:", ok.mean().round(3), pd.Series(ok).groupby(lab).mean().round(3).to_dict())
    for c in sorted(set(lab)):
        sheet(df[df.label == c], f"sheet_{c}.jpg")


if __name__ == "__main__":
    main()
