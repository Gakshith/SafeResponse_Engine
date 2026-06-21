import json
from pathlib import Path


def test_demo_corpus_has_breadth():
    data = json.loads(Path("data/demo_corpus.json").read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) >= 40, "corpus should have >= 40 articles"
    titles = set()
    for item in data:
        assert item["title"], "every article needs a title"
        assert len(item["text"]) > 400, f"article '{item['title']}' text too short"
        titles.add(item["title"])
    assert len(titles) == len(data), "article titles must be unique"
