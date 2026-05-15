from fastapi.testclient import TestClient

from saferesponse_engine.serving import api
from saferesponse_engine.serving.state_store import APIStateStore


class FakeMemory:
    def list_conversations(self, limit: int = 50):
        return [
            {
                "conversation_id": "abc",
                "title": "Demo chat",
                "preview": "chat answer",
                "updated_at": 123.0,
                "turn_count": 2,
                "message_count": 1,
            }
        ][:limit]

    def get_conversation(self, conversation_id: str):
        if conversation_id != "abc":
            return None
        return {
            "conversation_id": "abc",
            "turns": [
                {
                    "turn_id": 1,
                    "role": "user",
                    "text": "hi",
                    "timestamp": 122.0,
                    "metadata": {},
                },
                {
                    "turn_id": 2,
                    "role": "assistant",
                    "text": "chat answer",
                    "timestamp": 123.0,
                    "metadata": {},
                },
            ],
        }


class FakeEngine:
    memory = FakeMemory()

    def status(self):
        return {
            "status": "ok",
            "pipeline_busy": False,
            "final_output_exists": True,
            "final_output_path": "fake.json",
        }

    def load_final_output(self):
        return {"fake": "output"}

    def api_response(self, output):
        return {
            "query": "cached",
            "decision": "ACCEPT",
            "answer": "cached answer",
            "confidence": "HIGH",
            "confidence_color": "green",
            "source": "Demo Source (Wikipedia)",
            "risk_score": 0.1,
            "warnings": [],
            "risk_explanation": "No risk signals detected.",
            "decision_reason": "fake",
            "formatted_response": "cached answer",
            "pipeline_summary": {},
        }

    def query(self, query: str, run_pipeline: bool = False):
        return {
            "query": query,
            "decision": "ACCEPT",
            "answer": "query answer",
            "confidence": "HIGH",
            "confidence_color": "green",
            "source": "Demo Source (Wikipedia)",
            "risk_score": 0.1,
            "warnings": [],
            "risk_explanation": "No risk signals detected.",
            "decision_reason": "fake",
            "formatted_response": "query answer",
            "pipeline_summary": {"pipeline_run": run_pipeline},
        }

    def chat(
        self,
        message: str,
        conversation_id: str | None = None,
        run_pipeline: bool = True,
    ):
        return {
            "query": message,
            "decision": "DIRECT",
            "answer": "chat answer",
            "confidence": "HIGH",
            "confidence_color": "green",
            "source": None,
            "risk_score": 0.0,
            "warnings": [],
            "risk_explanation": "No retrieval was needed.",
            "decision_reason": "fake",
            "formatted_response": "chat answer",
            "pipeline_summary": {"pipeline_run": run_pipeline},
            "conversation_id": conversation_id or "fake-conversation",
            "mode": "single_turn",
            "intent": "small_talk",
        }


def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(api, "_engine", lambda: FakeEngine())
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_final_response_endpoint(monkeypatch):
    monkeypatch.setattr(api, "_engine", lambda: FakeEngine())
    client = TestClient(api.app)

    response = client.get("/v1/final-response")

    assert response.status_code == 200
    assert response.json()["decision"] == "ACCEPT"


def test_query_endpoint(monkeypatch):
    monkeypatch.setattr(api, "_engine", lambda: FakeEngine())
    client = TestClient(api.app)

    response = client.post(
        "/v1/query",
        json={"query": "Who was Abraham Lincoln?", "run_pipeline": False},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "query answer"


def test_chat_endpoint(monkeypatch):
    monkeypatch.setattr(api, "_engine", lambda: FakeEngine())
    client = TestClient(api.app)

    response = client.post(
        "/v1/chat",
        json={
            "message": "hi",
            "conversation_id": "abc",
            "run_pipeline": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["conversation_id"] == "abc"


def test_metrics_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_engine", lambda: FakeEngine())
    monkeypatch.setattr(api, "STATE_STORE", APIStateStore(tmp_path / "state.sqlite3"))
    client = TestClient(api.app)

    client.post("/v1/query", json={"query": "demo", "run_pipeline": False})
    response = client.get("/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["request_count"] == 1
    assert body["decision_counts"]["ACCEPT"] == 1
    assert body["average_latency_seconds"] is not None


def test_chat_job_endpoints(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_engine", lambda: FakeEngine())
    monkeypatch.setattr(api, "STATE_STORE", APIStateStore(tmp_path / "state.sqlite3"))
    client = TestClient(api.app)

    created = client.post(
        "/v1/chat/jobs",
        json={"message": "hi", "conversation_id": "job-abc", "run_pipeline": False},
    )
    assert created.status_code == 200
    job_id = created.json()["job_id"]

    fetched = client.get(f"/v1/chat/jobs/{job_id}")

    assert fetched.status_code == 200
    assert fetched.json()["status"] == "completed"
    assert fetched.json()["result"]["conversation_id"] == "job-abc"


def test_conversation_history_endpoints(monkeypatch):
    monkeypatch.setattr(api, "_engine", lambda: FakeEngine())
    client = TestClient(api.app)

    listed = client.get("/v1/conversations")
    loaded = client.get("/v1/conversations/abc")
    missing = client.get("/v1/conversations/missing")

    assert listed.status_code == 200
    assert listed.json()["conversations"][0]["title"] == "Demo chat"
    assert loaded.status_code == 200
    assert loaded.json()["turns"][1]["text"] == "chat answer"
    assert missing.status_code == 404


def test_api_key_protects_non_public_routes(monkeypatch):
    monkeypatch.setenv("SAFE_RESPONSE_API_KEY", "secret")
    monkeypatch.setattr(api, "_engine", lambda: FakeEngine())
    client = TestClient(api.app)

    assert client.get("/health").status_code == 200
    rejected = client.get("/metrics")
    accepted = client.get("/metrics", headers={"X-API-Key": "secret"})

    assert rejected.status_code == 401
    assert accepted.status_code == 200


def test_rate_limit_can_reject_excess_requests(monkeypatch):
    monkeypatch.setattr(api, "RATE_LIMIT_PER_MINUTE", 1)
    api.RATE_LIMIT_BUCKETS.clear()
    monkeypatch.setattr(api, "_engine", lambda: FakeEngine())
    client = TestClient(api.app)

    assert client.get("/health").status_code == 200
    assert client.get("/health").status_code == 429
