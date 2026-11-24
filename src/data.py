"""File names carry the capture time: YYYYMMDD_HHMMSS[_frame_N].jpg"""
import re
from datetime import datetime

CLASSES = ["Capsicum", "Carrot", "Cucumber", "Potato", "Raddish", "Squash"]
_NAME = re.compile(r"(\d{8}_\d{6})(?:_frame_(\d+))?\.jpg")


def parse_name(name: str) -> tuple[datetime, int | None]:
    m = _NAME.fullmatch(name)
    if m is None:
        raise ValueError(f"unexpected file name: {name}")
    frame = int(m.group(2)) if m.group(2) is not None else None
    return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S"), frame
