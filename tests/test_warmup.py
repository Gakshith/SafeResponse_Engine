from saferesponse_engine.components.chat_engine import SafeResponseChatEngine


def test_engine_has_warmup_method():
    assert hasattr(SafeResponseChatEngine, "warmup")
    assert callable(SafeResponseChatEngine.warmup)


def test_status_reports_warm_flag():
    engine = SafeResponseChatEngine()
    status = engine.status()
    assert "warm" in status
