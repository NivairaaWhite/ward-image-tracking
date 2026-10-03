"""
tracker.py
----------
Reverse-image-search integration via SerpApi's Google Lens API.

Requires the user's own SerpApi key (Settings panel or SERPAPI_API_KEY in
.env). SerpApi has a free tier; check https://serpapi.com/pricing for the
current limits.

Flow (per SerpApi's Image API + Google Lens API):
  1. POST the image to https://serpapi.com/image (multipart/form-data) to
     get a short-lived image_id, so a *local* file can be searched without
     hosting it publicly. The upload must be small (500 KB) and JPG/PNG/WebP.
  2. GET https://serpapi.com/search.json?engine=google_lens&image_id=...
     and read `visual_matches` from the response.

Note: SerpApi's `visual_matches` entries carry no "exact match" flag (exact
matches are a separate `type=` search), so Ward does not show one.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import requests
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

SERPAPI_IMAGE_UPLOAD_URL = "https://serpapi.com/image"
SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"
MAX_UPLOAD_BYTES = 500 * 1024  # SerpApi's documented Image API limit
MAX_UPLOAD_DIMENSION = 2048
REQUEST_TIMEOUT = 30


class TrackerError(Exception):
    """Any SerpApi problem -- caught by the caller and shown to the user as
    a clear message, never a stack trace."""


@dataclass
class Match:
    title: str = ""
    link: str = ""
    source: str = ""
    thumbnail: str = ""
    image: str = ""


@dataclass
class SearchResult:
    matches: list[Match] = field(default_factory=list)


def is_configured(api_key: str | None) -> bool:
    return bool(api_key and api_key.strip())


def _compress_for_upload(image: Image.Image) -> bytes:
    """Re-encode to JPEG under SerpApi's 500 KB limit.

    Caps the longest side first (so huge photos start small), then lowers
    quality, and if that still doesn't fit, shrinks 20% and tries again.
    """
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    rgb.thumbnail((MAX_UPLOAD_DIMENSION, MAX_UPLOAD_DIMENSION))
    for _ in range(8):
        for quality in (85, 75, 65, 55):
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=quality, optimize=True)
            if buf.tell() <= MAX_UPLOAD_BYTES:
                return buf.getvalue()
        rgb = rgb.resize(
            (max(1, int(rgb.width * 0.8)), max(1, int(rgb.height * 0.8))), Image.LANCZOS
        )
    raise TrackerError("Could not shrink this image under SerpApi's 500 KB upload limit.")


def _json(resp: requests.Response) -> dict:
    try:
        data = resp.json()
    except ValueError:
        raise TrackerError(f"SerpApi returned a non-JSON response (HTTP {resp.status_code}).") from None
    if not isinstance(data, dict):
        raise TrackerError("SerpApi returned an unexpected response.")
    return data


def _upload_image(image_bytes: bytes, api_key: str) -> str:
    try:
        resp = requests.post(
            SERPAPI_IMAGE_UPLOAD_URL,
            files={"image": ("upload.jpg", image_bytes, "image/jpeg")},
            data={"api_key": api_key},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        raise TrackerError(f"Could not reach SerpApi to upload the image: {e}") from e

    data = _json(resp)
    if "error" in data:
        raise TrackerError(f"SerpApi rejected the image upload: {data['error']}")
    image_id = data.get("image_id")
    if not image_id:
        raise TrackerError("SerpApi did not return an image_id for the upload.")
    return image_id


def _parse_visual_matches(data: dict) -> SearchResult:
    if "error" in data:
        raise TrackerError(f"SerpApi search failed: {data['error']}")

    matches: list[Match] = []
    seen: set[str] = set()
    for m in data.get("visual_matches", []) or []:
        link = (m.get("link") or "").strip()
        if not link or link in seen:  # unusable or repeated within one result set
            continue
        seen.add(link)
        matches.append(
            Match(
                title=m.get("title", "") or "",
                link=link,
                source=m.get("source", "") or "",
                thumbnail=m.get("thumbnail", "") or "",
                image=m.get("image", "") or "",
            )
        )
    return SearchResult(matches=matches)


def search_by_upload(image: Image.Image, api_key: str) -> SearchResult:
    """Upload `image` to SerpApi and run a Google Lens search on it."""
    if not is_configured(api_key):
        raise TrackerError("No SerpApi key configured. Add one under Settings.")

    image_id = _upload_image(_compress_for_upload(image), api_key)

    try:
        resp = requests.get(
            SERPAPI_SEARCH_URL,
            params={"engine": "google_lens", "image_id": image_id, "type": "visual_matches", "api_key": api_key},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        raise TrackerError(f"Could not reach SerpApi to run the search: {e}") from e

    return _parse_visual_matches(_json(resp))
