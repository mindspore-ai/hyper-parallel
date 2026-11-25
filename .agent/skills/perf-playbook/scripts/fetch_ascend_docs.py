# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Fetch attributed Ascend API text, or retrieve candidate cards without network access."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request


CATALOG = Path(__file__).resolve().parents[1] / "references/ascend-fusion/catalog.json"


class DocumentText(HTMLParser):
    """Extract visible documentation while preserving table and code boundaries."""

    def __init__(self) -> None:
        """Initialize visible text and hidden element depth."""
        super().__init__()
        self.fragments = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        """Retain structural boundaries while ignoring executable page elements."""
        if tag in ("script", "style", "noscript"):
            self.hidden += 1
        if tag in ("p", "div", "tr", "li", "pre", "h1", "h2", "h3", "h4", "br"):
            self.fragments.append("\n")
        if tag in ("td", "th"):
            self.fragments.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        """Resume visible text after a hidden element closes."""
        if tag in ("script", "style", "noscript"):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data: str) -> None:
        """Keep text only from visible documentation nodes."""
        if not self.hidden:
            self.fragments.append(data)


def extract_document(raw: bytes, symbol: str) -> str:
    """Reject challenge/error pages even when their HTTP status is 200."""
    html = raw.decode("utf-8")
    if any(marker in html for marker in ("HuaweiCloudWAF", "Please Enable JavaScript", "Access Denied")):
        raise ValueError("Server returned a challenge/error page, not documentation")
    parser = DocumentText()
    parser.feed(html)
    text = re.sub(r"\n[ \t]*\n+", "\n\n", "".join(parser.fragments)).strip()
    title = re.search(r"<h1\b[^>]*>(.*?)</h1>", html, re.DOTALL)
    if not title or symbol not in title.group(1) or len(text) < 300:
        raise ValueError(f"Missing documentation title/body for {symbol}")
    return text + "\n"


def fetch_entry(entry: dict, directory: Path) -> dict:
    """Save an immutable content-addressed text snapshot and its provenance."""
    url = entry["content_url"]
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "www.hiascend.com":
        raise ValueError("Only HTTPS official www.hiascend.com documentation is accepted")
    request = urllib.request.Request(url, headers={"User-Agent": "HyperParallel-doc-cache/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if urllib.parse.urlparse(response.url).hostname != "www.hiascend.com":
            raise ValueError("Documentation redirected outside official host")
        raw = response.read(4_000_001)
        resolved = response.url
    if len(raw) > 4_000_000:
        raise ValueError("Document exceeds 4 MB limit")
    text = extract_document(raw, entry["symbol"])
    digest = hashlib.sha256(text.encode()).hexdigest()
    name = f"{entry['id']}-{digest[:12]}.txt"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(text.encode("utf-8"))
    return {**entry, "fetched_utc": datetime.now(timezone.utc).isoformat(),
            "resolved_url": resolved, "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "text_sha256": digest, "text_file": name, "bytes": len(raw)}


def main(argv: list[str] | None = None) -> int:
    """Fetch a catalog or search cached cards; failed refreshes retain older evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--query", help="Offline keyword lookup; returns candidates, not compatibility verdicts")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    entries = json.loads(args.catalog.read_text(encoding="utf-8"))["entries"]
    if args.query is not None:
        words = args.query.lower().split()
        hits = [entry for entry in entries if any(word in json.dumps(entry).lower() for word in words)]
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return 0
    directory = args.out or Path.home() / ".cache/hyper_parallel/ascend-fusion"
    records, errors = [], []
    for entry in entries:
        try:
            records.append(fetch_entry(entry, directory))
        except (OSError, ValueError) as exc:
            errors.append({"id": entry["id"], "error": str(exc)})
    directory.mkdir(parents=True, exist_ok=True)
    result = {"schema": "ascend-doc-fetch/v1", "documents": records, "errors": errors}
    # A failed refresh must not erase a previously useful manifest.
    path = directory / ("failed-refresh.json" if errors else "manifest.json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fetched": len(records), "errors": errors, "manifest": str(path)}))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
