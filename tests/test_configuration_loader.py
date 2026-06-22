from saferesponse_engine.config.configuration import ConfigurationManager


def test_generation_config_loads_with_force_answer_flag():
    config = ConfigurationManager().get_generation_layer_config()
    assert isinstance(config.force_answer, bool)
    assert config.model_name
    assert config.max_new_tokens > 0
