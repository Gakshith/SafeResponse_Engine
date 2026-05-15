from saferesponse_engine.serving.state_store import APIStateStore


def test_state_store_persists_jobs(tmp_path):
    db_path = tmp_path / "state.sqlite3"
    store = APIStateStore(db_path)

    store.create_job("job-1", "query", "Who was Abraham Lincoln?")
    store.update_job(
        "job-1",
        status="completed",
        completed_at=123.0,
        result={"decision": "ACCEPT"},
    )

    restored = APIStateStore(db_path).get_job("job-1")

    assert restored["status"] == "completed"
    assert restored["result"]["decision"] == "ACCEPT"


def test_state_store_metrics_summary(tmp_path):
    store = APIStateStore(tmp_path / "state.sqlite3")

    store.record_metric(
        latency_seconds=0.25,
        decision="ACCEPT",
        risk_score=0.1,
        success=True,
    )
    store.record_metric(
        latency_seconds=0.5,
        decision="REJECT",
        risk_score=1.0,
        success=True,
    )

    summary = store.metrics_summary()

    assert summary["request_count"] == 2
    assert summary["decision_counts"] == {"REJECT": 1, "ACCEPT": 1}
    assert summary["average_latency_seconds"] == 0.375
    assert summary["risk_score_stats"]["max"] == 1.0
