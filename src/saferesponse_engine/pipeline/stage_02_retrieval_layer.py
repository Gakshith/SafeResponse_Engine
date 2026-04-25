from src.saferesponse_engine.config.configuration import ConfigurationManager
from src.saferesponse_engine.components.retrieval_layer import RetrievalLayer


STAGE_NAME = "Retrieval_Layer_STAGE"


class RetrievalLayerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        cm = ConfigurationManager()
        retrieval_config = cm.get_retrieval_layer_config()
        retrieval_layer = RetrievalLayer(config=retrieval_config)
        retrieval_layer.retrieve()
