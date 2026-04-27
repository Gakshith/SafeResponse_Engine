from src.saferesponse_engine.components.trace_collection_layer import TraceCollectionLayer
from src.saferesponse_engine.config.configuration import ConfigurationManager


STAGE_NAME = "Trace_Collection_Layer_STAGE"


class TraceCollectionLayerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        cm = ConfigurationManager()
        trace_collection_config = cm.get_trace_collection_config()
        trace_collection_layer = TraceCollectionLayer(config=trace_collection_config)
        trace_collection_layer.collect()
