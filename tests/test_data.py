from datetime import datetime

import pytest

from src.data import parse_name


def test_video_frame():
    assert parse_name("20260101_132853_frame_2.jpg") == (datetime(2026, 1, 1, 13, 28, 53), 2)


def test_still_photo_has_no_frame():
    assert parse_name("20251113_205619.jpg") == (datetime(2025, 11, 13, 20, 56, 19), None)


def test_unknown_name_is_an_error():
    with pytest.raises(ValueError):
        parse_name("IMG_0001.jpg")
