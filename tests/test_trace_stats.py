from saferesponse_engine.components.trace_stats import summarize_logprobs


def test_summarize_logprobs_basic():
    s = summarize_logprobs([-0.5, -1.5, 0.0])
    assert s["mean_logprob"] == round(-2.0 / 3, 6)
    assert s["min_logprob"] == -1.5
    assert s["sequence_score"] == -2.0


def test_summarize_logprobs_empty():
    assert summarize_logprobs([]) == {
        "mean_logprob": 0.0,
        "min_logprob": 0.0,
        "sequence_score": 0.0,
    }
