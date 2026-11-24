import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.data import CLASSES, load_photo

ROOT = Path(__file__).resolve().parents[1]


def run(inp, out, *extra, cwd=None):
    return subprocess.run([sys.executable, str(ROOT / "predict.py"), "--input", str(inp), "--output", str(out), *extra],
                          capture_output=True, text=True, encoding="utf-8", cwd=cwd or ROOT)


def rows(out):
    with open(out, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def noise(seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 255, (80, 120, 3), dtype=np.uint8))


def test_mixed_folder(tmp_path):
    d = tmp_path / "фото"
    d.mkdir()
    noise().save(d / "огурец.JPG", "JPEG")
    noise(1).convert("RGBA").save(d / "alpha.png")
    noise(2).convert("L").save(d / "gray.jpg")
    noise(3).convert("P").save(d / "palette.png")
    (d / "broken.jpg").write_bytes(b"not an image")
    good = (d / "gray.jpg").read_bytes()
    (d / "truncated.jpg").write_bytes(good[: len(good) // 2])
    (d / "notes.txt").write_text("x")
    (d / "sub").mkdir()
    out = tmp_path / "preds.csv"
    r = run(d, out, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    table = rows(out)
    assert table[0] == ["filename", "label", "confidence"]
    assert [x[0] for x in table[1:]] == ["alpha.png", "gray.jpg", "palette.png", "огурец.JPG"]
    assert all(x[1] in CLASSES and 0 <= float(x[2]) <= 1 for x in table[1:])
    for name in ("broken.jpg", "truncated.jpg", "notes.txt", "sub"):
        assert name in r.stderr


def test_empty_folder_gives_header_only(tmp_path):
    (tmp_path / "e").mkdir()
    out = tmp_path / "p.csv"
    assert run(tmp_path / "e", out).returncode == 0
    assert rows(out) == [["filename", "label", "confidence"]]


def test_missing_folder_is_a_clear_error(tmp_path):
    r = run(tmp_path / "nope", tmp_path / "p.csv")
    assert r.returncode != 0 and "nope" in r.stderr and "Traceback" not in r.stderr


def test_exif_rotation_is_applied(tmp_path):
    im = Image.new("RGB", (40, 20))
    exif = im.getexif()
    exif[274] = 6
    im.save(tmp_path / "r.jpg", exif=exif)
    assert load_photo(tmp_path / "r.jpg").size == (20, 40)


def test_transparent_pixels_become_white(tmp_path):
    Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(tmp_path / "t.png")
    assert load_photo(tmp_path / "t.png").getpixel((0, 0)) == (255, 255, 255)


def test_unknown_threshold_is_off_by_default_and_optional(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    noise().save(d / "a.png")
    plain, strict = tmp_path / "p.csv", tmp_path / "s.csv"
    assert run(d, plain).returncode == 0
    r = run(d, strict, "--unknown-threshold", "1.01")
    assert r.returncode == 0, r.stderr
    assert rows(plain)[1][1] in CLASSES
    assert rows(strict)[1][1] == "unknown"
    assert rows(plain)[1][2] == rows(strict)[1][2]


def test_output_folder_is_created(tmp_path):
    (tmp_path / "in").mkdir()
    out = tmp_path / "new" / "dir" / "p.csv"
    r = run(tmp_path / "in", out)
    assert r.returncode == 0, r.stderr
    assert rows(out) == [["filename", "label", "confidence"]]


def test_unwritable_output_is_a_clear_error(tmp_path):
    (tmp_path / "in").mkdir()
    r = run(tmp_path / "in", tmp_path)  # a folder instead of a file
    assert r.returncode != 0 and "cannot write" in r.stderr and "Traceback" not in r.stderr


def test_16_bit_gray_is_scaled_not_clipped(tmp_path):
    Image.fromarray(np.full((4, 4), 32768, dtype=np.uint16)).save(tmp_path / "g.png")
    assert abs(load_photo(tmp_path / "g.png").getpixel((0, 0))[0] - 128) <= 1
