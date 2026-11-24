"""Predict a vegetable for every photo in a folder.

python predict.py --input path/to/folder --output preds.csv
Works on CPU and without internet.
"""
import argparse
import csv
import sys
from pathlib import Path

import torch

from src.data import CLASSES, load_photo
from src.model import build_model, get_transforms

WEIGHTS = Path(__file__).resolve().parent / "weights" / "model.pt"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def load_model() -> torch.nn.Module:
    net = build_model(len(CLASSES), pretrained=False)
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    return net.eval()


def predict_folder(input_dir: Path, unknown_threshold: float = 0.0) -> list[tuple[str, str, float]]:
    net, transform, result = load_model(), get_transforms(False), []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            warn(f"skipped, not a photo: {path.name}")
            continue
        try:
            x = transform(load_photo(path))[None]
        except Exception as e:  # any unreadable photo must not stop the others
            warn(f"skipped, cannot read {path.name}: {e}")
            continue
        with torch.inference_mode():
            probs = net(x).softmax(1)[0]
        k = int(probs.argmax())
        conf = float(probs[k])
        result.append((path.name, CLASSES[k] if conf >= unknown_threshold else "unknown", conf))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--unknown-threshold", type=float, default=0.0,
                    help="write `unknown` below this confidence; off by default, see REPORT.md (0.79)")
    a = ap.parse_args()
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not a.input.is_dir():
        sys.exit(f"error: input folder not found: {a.input}")
    try:  # before the long part, so a locked or wrong path fails at once
        a.output.parent.mkdir(parents=True, exist_ok=True)
        f = open(a.output, "w", newline="", encoding="utf-8")
    except OSError as e:
        sys.exit(f"error: cannot write {a.output}: {e.strerror}")
    with f:
        rows = predict_folder(a.input, a.unknown_threshold)
        w = csv.writer(f)
        w.writerow(["filename", "label", "confidence"])
        w.writerows((n, l, f"{c:.4f}") for n, l, c in rows)
    print(f"{len(rows)} photos -> {a.output}")


if __name__ == "__main__":
    main()
