"""Unpack only Dataset/ and record the author's split as a CSV.

Usage: python src/split_list.py path/to/archive.zip
"""
import csv
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(zip_path: str) -> None:
    out = ROOT / "data" / "Dataset"
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            parts = info.filename.split("/")
            label, name = parts[-2], parts[-1]
            if parts[0] == "Dataset":
                target = out / label / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or target.stat().st_size != info.file_size:
                    target.write_bytes(z.read(info))
            else:
                rows.append((label, name, parts[2]))
    with open(ROOT / "data" / "author_split.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "name", "author_split"])
        w.writerows(sorted(rows))


if __name__ == "__main__":
    main(sys.argv[1])
