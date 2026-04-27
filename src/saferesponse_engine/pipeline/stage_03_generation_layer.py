from src.saferesponse_engine.components.generation_layer import GenerationLayer
from src.saferesponse_engine.config.configuration import ConfigurationManager


STAGE_NAME = "Generation_Layer_STAGE"


class GenerationLayerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        cm = ConfigurationManager()
        generation_config = cm.get_generation_layer_config()
        generation_layer = GenerationLayer(config=generation_config)
        generation_layer.generate()
