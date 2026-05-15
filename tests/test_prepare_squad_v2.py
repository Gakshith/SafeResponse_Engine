from scripts.prepare_squad_v2 import _answers_texts, _balanced_sample
from scripts.finetune_model import register_training_run


def test_answers_texts_handles_empty_squad_answers():
    assert _answers_texts({"text": [], "answer_start": []}) == []


def test_answers_texts_strips_non_empty_answers():
    assert _answers_texts({"text": [" Denver Broncos ", ""], "answer_start": [0]}) == [
        "Denver Broncos"
    ]


def test_balanced_sample_prefers_answerable_and_unanswerable_mix():
    records = [
        {"squad_id": f"a-{index}", "is_answerable": True}
        for index in range(4)
    ] + [
        {"squad_id": f"u-{index}", "is_answerable": False}
        for index in range(4)
    ]

    selected = _balanced_sample(records=records, max_records=4, seed=1)

    assert len(selected) == 4
    assert sum(1 for record in selected if record["is_answerable"]) == 2
    assert sum(1 for record in selected if not record["is_answerable"]) == 2


def test_register_training_run_appends_metadata(tmp_path):
    registry_path = tmp_path / "registry.json"

    register_training_run(registry_path, {"run_name": "smoke"})
    register_training_run(registry_path, {"run_name": "candidate"})

    assert '"run_name": "smoke"' in registry_path.read_text(encoding="utf-8")
    assert '"run_name": "candidate"' in registry_path.read_text(encoding="utf-8")
