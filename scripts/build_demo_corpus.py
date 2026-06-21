"""Build the controlled demo corpus for SafeResponse Engine.

Fetches plain-text extracts for a curated, broad set of Wikipedia articles and
writes them to ``data/demo_corpus.json`` as ``[{"title", "text"}, ...]``.

The corpus is intentionally controlled (~45 articles across history, science,
geography, technology, sports, arts, biology, and space) so that the engine can
demonstrate a clean supported/unsupported split: questions about these topics
should be answerable and grounded, while everything else should be rejected.

The generated file is committed to the repo so the demo runs fully offline;
re-run this script (network required) only to regenerate the corpus.

Usage:
    venv/bin/python scripts/build_demo_corpus.py [--max-chars 3000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests


# Curated, broad set of well-known topics with stable Wikipedia titles.
CURATED_TITLES: list[str] = [
    # History / people
    "Abraham Lincoln",
    "Mahatma Gandhi",
    "Cleopatra",
    "Napoleon",
    "Nelson Mandela",
    "World War II",
    "The Renaissance",
    "Industrial Revolution",
    # Science / physics
    "Albert Einstein",
    "Isaac Newton",
    "Theory of relativity",
    "Quantum mechanics",
    "Photosynthesis",
    "DNA",
    "Periodic table",
    "Electricity",
    # Space / astronomy
    "Solar System",
    "Black hole",
    "Moon",
    "Mars",
    "Apollo 11",
    "Hubble Space Telescope",
    # Geography
    "Mount Everest",
    "Amazon rainforest",
    "Sahara",
    "Nile",
    "Pacific Ocean",
    "Japan",
    "France",
    "Brazil",
    # Technology / computing
    "Internet",
    "Artificial intelligence",
    "Computer",
    "Linux",
    "World Wide Web",
    "Machine learning",
    # Arts / culture
    "Leonardo da Vinci",
    "William Shakespeare",
    "The Beatles",
    "Mona Lisa",
    # Sports
    "FIFA World Cup",
    "Olympic Games",
    "Cricket",
    # Biology / nature
    "Human brain",
    "Honey bee",
    "Coral reef",
]

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "SafeResponseEngine-corpus-builder/1.0 (research demo)"


def fetch_extract(title: str, max_chars: int, session: requests.Session) -> str:
    """Return the plain-text extract for a Wikipedia article title."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "titles": title,
        "format": "json",
        "formatversion": "2",
    }
    # Retry with exponential backoff on rate limiting (429) / transient errors.
    last_exc: Exception | None = None
    for attempt in range(6):
        try:
            response = session.get(params=params, url=WIKI_API, timeout=30)
            if response.status_code == 429:
                wait = 2 ** attempt
                print(f"    (429 on {title!r}; backing off {wait}s)", file=sys.stderr)
                time.sleep(wait)
                continue
            response.raise_for_status()
            break
        except requests.RequestException as exc:  # pragma: no cover - network
            last_exc = exc
            time.sleep(2 ** attempt)
    else:
        raise last_exc if last_exc else requests.RequestException(f"429 for {title}")
    pages = response.json().get("query", {}).get("pages", [])
    if not pages:
        return ""
    text = (pages[0].get("extract") or "").strip()
    if max_chars and len(text) > max_chars:
        # Trim to the last sentence boundary at or before max_chars.
        cut = text[:max_chars]
        last_stop = max(cut.rfind(". "), cut.rfind(".\n"))
        text = cut[: last_stop + 1] if last_stop > 400 else cut
    return text


def build(output_path: Path, max_chars: int) -> int:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    corpus: list[dict[str, str]] = []
    seen: set[str] = set()
    for title in CURATED_TITLES:
        if title in seen:
            continue
        try:
            text = fetch_extract(title, max_chars=max_chars, session=session)
        except requests.RequestException as exc:  # pragma: no cover - network
            print(f"  ! failed {title!r}: {exc}", file=sys.stderr)
            continue
        if len(text) <= 400:
            print(f"  ! skipping {title!r}: extract too short", file=sys.stderr)
            continue
        corpus.append({"title": title, "text": text})
        seen.add(title)
        print(f"  + {title} ({len(text)} chars)")
        time.sleep(1.0)  # be polite to the API

    # Never clobber a good corpus with a worse one: only write if we collected a
    # usable number of articles (and at least as many as already on disk).
    existing = 0
    if output_path.exists():
        try:
            existing = len(json.loads(output_path.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            existing = 0
    if len(corpus) < 40 and len(corpus) <= existing:
        print(
            f"\nCollected only {len(corpus)} articles (disk has {existing}); "
            "leaving the existing corpus untouched.",
            file=sys.stderr,
        )
        return existing

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(corpus, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote {len(corpus)} articles to {output_path}")
    return len(corpus)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the demo corpus.")
    parser.add_argument(
        "--output",
        default="data/demo_corpus.json",
        help="Output JSON path (default: data/demo_corpus.json).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=3000,
        help="Max characters of extract per article (default: 3000).",
    )
    args = parser.parse_args()
    count = build(Path(args.output), max_chars=args.max_chars)
    if count < 40:
        print(
            f"WARNING: only {count} articles fetched (< 40). Check network access.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
