import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABLATION_SCRIPT = PROJECT_ROOT / "scripts" / "run_ablation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_ablation", ABLATION_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_configs_has_expected_named_runs():
    module = _load_module()
    configs = module.build_configs()
    for name in ("all_off", "all_on", "logprob_only", "grounding_only", "consistency_only"):
        assert name in configs, f"missing ablation config {name!r}"
        assert isinstance(configs[name], dict)


def test_all_off_disables_every_signal():
    module = _load_module()
    all_off = module.build_configs()["all_off"]
    assert all_off["enable_halluguard"] is False
    assert all_off["enable_grounding_score"] is False
    assert all_off["enable_consistency_score"] is False


def test_classify_example_maps_decision_to_support():
    module = _load_module()
    assert module.classify_example({"expected_decision": "ACCEPT"}) is True
    assert module.classify_example({"expected_decision": "REJECT"}) is False
    # Small-talk / direct examples are not hallucination cases -> excluded (None).
    assert module.classify_example({"expected_decision": "DIRECT"}) is None
