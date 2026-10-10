from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
import hashlib
import html
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2
import numpy as np

from set_master.models import BackgroundSplit
from set_master.paths import BACKGROUNDS_DIR


DEFAULT_QUERIES = (
    '"empty table"', '"table top"', '"tabletop"', '"wood texture"',
    '"wooden surface"', '"desk surface"', '"wood grain"', '"tablecloth"',
    '"fabric texture"', '"stone texture"', '"marble texture"', '"laminate texture"',
)
COMMONS_API = "https://commons.wikimedia.org/w/api.php"


@dataclass
class BackgroundRecord:
    file_name: str
    source_id: int
    title: str
    source_url: str
    download_url: str
    source_sha1: str
    pixel_sha256: str
    author: str
    license: str
    license_url: str
    query: str
    width: int
    height: int
    split: str
    approved: bool = False


def load_backgrounds(root: Path) -> list[BackgroundRecord]:
    path = root / "manifest.json"
    if not path.exists():
        return []
    return [BackgroundRecord(**record) for record in json.loads(path.read_text())["images"]]


def save_backgrounds(root: Path, records: list[BackgroundRecord]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "manifest.tmp"
    temporary.write_text(json.dumps({"source": "Wikimedia Commons", "images": [asdict(r) for r in records]},
                                    indent=2) + "\n")
    temporary.replace(root / "manifest.json")


def _read_url(url: str, user_agent: str) -> bytes:
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": user_agent}), timeout=30) as response:
                payload = response.read(20 * 1024 * 1024 + 1)
                if len(payload) > 20 * 1024 * 1024:
                    raise ValueError("Response exceeds 20 MiB")
                return payload
        except HTTPError as error:
            if attempt == 2 or error.code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(2 ** (attempt + 1))
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("Download retries exhausted")


def _search(query: str, user_agent: str, max_results: int) -> Iterator[dict[str, Any]]:
    # API responses are the dynamic boundary; stored records have a fixed schema.
    continuation: dict[str, Any] = {}
    retrieved = 0
    while retrieved < max_results:
        parameters = {
            "action": "query", "format": "json", "formatversion": 2,
            "generator": "search", "gsrsearch": query, "gsrnamespace": 6,
            "gsrlimit": min(10, max_results - retrieved), "prop": "imageinfo",
            "iiprop": "url|size|mime|sha1|extmetadata", "iiurlwidth": 1440,
            "iiextmetadatafilter": "Artist|LicenseShortName|LicenseUrl|ImageDescription",
            **continuation,
        }
        result = json.loads(_read_url(COMMONS_API + "?" + urlencode(parameters), user_agent))
        if "error" in result:
            raise RuntimeError(f"Commons search failed: {result['error']}")
        pages = result.get("query", {}).get("pages", [])
        yield from pages
        retrieved += len(pages)
        if "continue" not in result or not pages:
            break
        continuation = result["continue"]
        time.sleep(0.5)


def download_backgrounds(
    count: int = 1000, root: Path = BACKGROUNDS_DIR, queries: tuple[str, ...] = DEFAULT_QUERIES,
    user_agent: str = "SetMasterBackgroundDownloader/1.0", min_side: int = 480,
    max_results_per_query: int = 500, progress: Callable[[int, int], None] | None = None,
) -> list[BackgroundRecord]:
    """Download deduplicated candidates, retaining attribution and review decisions on reruns."""
    if count < 1 or min_side < 1 or max_results_per_query < 1 or not queries:
        raise ValueError("Require positive download limits and at least one query")
    (root / "images").mkdir(parents=True, exist_ok=True)
    records = load_backgrounds(root)
    missing = [record.file_name for record in records if not (root / record.file_name).is_file()]
    if missing:
        raise FileNotFoundError(f"Manifest images are missing: {missing[:5]}")
    source_hashes = {r.source_sha1 for r in records}
    pixel_hashes = {r.pixel_sha256 for r in records}
    failures: list[dict[str, str]] = []
    for query in queries:
        if len(records) >= count:
            break
        for page in _search(query, user_agent, max_results_per_query):
            if len(records) >= count:
                break
            information = page.get("imageinfo", [])
            if not information:
                continue
            info = information[0]
            if info.get("mime") not in ("image/jpeg", "image/png", "image/webp"):
                continue
            if min(info["width"], info["height"]) < min_side or info["sha1"] in source_hashes:
                continue
            metadata = {name: html.unescape(re.sub(r"<[^>]*>", "", value.get("value", "")))
                        for name, value in info.get("extmetadata", {}).items()}
            license_name = metadata.get("LicenseShortName", "")
            if not license_name:
                continue
            url = info.get("thumburl", info["url"])
            try:
                payload = _read_url(url, user_agent)
                image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None or min(image.shape[:2]) < min_side:
                    continue
            except (URLError, TimeoutError, ValueError, cv2.error) as error:
                failures.append({"url": url, "error": str(error)})
                continue
            pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
            if pixel_hash in pixel_hashes:
                continue
            height, width = image.shape[:2]
            scale = min(1.0, 1440 / max(height, width))
            if scale < 1:
                image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            filename = f"images/{len(records) + 1:06d}.jpg"
            if not cv2.imwrite(str(root / filename), image):
                raise OSError(f"Could not save {root / filename}")
            # A source image always belongs to one partition, including on resumed downloads.
            split = BackgroundSplit.VALIDATION if int(pixel_hash[:8], 16) % 10 == 0 else BackgroundSplit.TRAIN
            records.append(BackgroundRecord(
                filename, page["pageid"], page["title"], info["descriptionurl"], url,
                info["sha1"], pixel_hash, metadata.get("Artist", ""), license_name, metadata.get("LicenseUrl", ""),
                query, image.shape[1], image.shape[0], split.value,
            ))
            source_hashes.add(info["sha1"])
            pixel_hashes.add(pixel_hash)
            save_backgrounds(root, records)
            if progress is not None:
                progress(len(records), count)
            time.sleep(0.25)
    save_backgrounds(root, records)
    (root / "download_failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    print(f"Downloaded {len(records)}/{count} candidates to {root}; approved: {sum(r.approved for r in records)}")
    return records


def crop_background(image: np.ndarray, width: int, height: int, x_fraction: float, y_fraction: float) -> np.ndarray:
    """Cover the canvas without stretching the surface texture."""
    source_height, source_width = image.shape[:2]
    scale = max(width / source_width, height / source_height)
    resized = cv2.resize(image, (max(width, round(source_width * scale)), max(height, round(source_height * scale))))
    x = round((resized.shape[1] - width) * x_fraction)
    y = round((resized.shape[0] - height) * y_fraction)
    return resized[y:y + height, x:x + width].copy()
