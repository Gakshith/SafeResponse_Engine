from src.saferesponse_engine.components.verification_layer import VerificationLayer
from src.saferesponse_engine.config.configuration import ConfigurationManager


STAGE_NAME = "Verification_Layer_STAGE"


class VerificationLayerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        cm = ConfigurationManager()
        verification_config = cm.get_verification_config()
        verification_layer = VerificationLayer(config=verification_config)
        verification_layer.verify()
