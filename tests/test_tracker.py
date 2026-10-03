import os

import pytest
from PIL import Image

from core import tracker


def test_compress_noisy_large_image_fits_limit():
    noisy = Image.frombytes("RGB", (4000, 3000), os.urandom(4000 * 3000 * 3))
    assert len(tracker._compress_for_upload(noisy)) <= tracker.MAX_UPLOAD_BYTES


def test_compress_handles_rgba_and_tiny_images():
    assert tracker._compress_for_upload(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))


def test_parse_dedupes_and_drops_linkless():
    data = {"visual_matches": [
        {"title": "a", "link": "http://a"},
        {"title": "a again", "link": "http://a"},
        {"title": "no link"},
        {"title": "b", "link": "http://b", "source": "B"},
    ]}
    assert [m.link for m in tracker._parse_visual_matches(data).matches] == ["http://a", "http://b"]


def test_parse_raises_on_serpapi_error():
    with pytest.raises(tracker.TrackerError):
        tracker._parse_visual_matches({"error": "Invalid API key"})


def test_search_requires_key():
    with pytest.raises(tracker.TrackerError):
        tracker.search_by_upload(Image.new("RGB", (4, 4)), "")
