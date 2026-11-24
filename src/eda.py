"""Per-image scan: size, EXIF, brightness, sharpness, MD5, pHash, embedding.

Images are processed one by one to fit in 8 GB RAM.
Writes reports/eda/images.csv and reports/eda/features.npz
"""
import hashlib
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.data import parse_name

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "eda"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / "data" / "Dataset").rglob("*.jpg"))
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    net = mobilenet_v3_small(weights=weights).eval()
    net.classifier = torch.nn.Identity()
    prep = weights.transforms()
    rows, hashes, embs = [], [], []
    for i, p in enumerate(files):
        raw = p.read_bytes()
        time, frame = parse_name(p.name)
        row = dict(path=p.relative_to(ROOT / "data" / "Dataset").as_posix(), label=p.parent.name,
                   time=time, frame=frame, bytes=len(raw), md5=hashlib.md5(raw).hexdigest(), error="")
        try:
            with Image.open(p) as im:
                row["orientation"] = im.getexif().get(274, 0)
                im = ImageOps.exif_transpose(im).convert("RGB")
        except OSError as e:
            row["error"] = str(e)
            rows.append(row)
            hashes.append(np.zeros(64, bool))
            embs.append(np.zeros(576, np.float32))
            continue
        row["width"], row["height"] = im.size
        small = im.resize((256, 256))
        g = np.asarray(small.convert("L"), dtype=np.float32)
        row["brightness"] = g.mean()
        row["sharpness"] = (4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]).var()
        hashes.append(imagehash.phash(im).hash.ravel())
        with torch.inference_mode():
            embs.append(net(prep(im)[None])[0].numpy())
        rows.append(row)
        if i % 500 == 0:
            print(i, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "images.csv", index=False)
    np.savez_compressed(OUT / "features.npz", phash=np.array(hashes), emb=np.array(embs))
    print("done", len(rows))


if __name__ == "__main__":
    main()
