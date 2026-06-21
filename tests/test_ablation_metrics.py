from saferesponse_engine.components.ablation_metrics import confusion


def test_confusion_counts():
    # records: (expected_supported: bool, decided_accept: bool)
    records = [(True, True), (True, False), (False, True), (False, False), (False, False)]
    m = confusion(records)
    assert m["true_accept"] == 1
    assert m["false_reject"] == 1   # supported but rejected
    assert m["false_accept"] == 1   # unsupported but accepted
    assert m["true_reject"] == 2
    assert m["total"] == 5
    assert round(m["false_accept_rate"], 3) == round(1 / 3, 3)  # 1 of 3 unsupported
    assert round(m["false_reject_rate"], 3) == round(1 / 2, 3)  # 1 of 2 supported


def test_confusion_handles_empty():
    m = confusion([])
    assert m["total"] == 0
    assert m["false_accept_rate"] == 0.0
    assert m["false_reject_rate"] == 0.0
