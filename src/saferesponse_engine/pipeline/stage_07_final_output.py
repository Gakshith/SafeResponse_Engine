from saferesponse_engine import logger
from saferesponse_engine.components.final_output import FinalOutputLayer
from saferesponse_engine.config.configuration import ConfigurationManager


STAGE_NAME = "Final Output Layer"


class FinalOutputPipeline:
    def __init__(self):
        pass

    def main(self):
        cm = ConfigurationManager()
        output_config = cm.get_final_output_config()
        final_output_layer = FinalOutputLayer(config=output_config)
        final_output_layer.generate()


if __name__ == "__main__":
    try:
        logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
        pipeline = FinalOutputPipeline()
        pipeline.main()
        logger.info(">>>>>> stage %s completed <<<<<<", STAGE_NAME)
    except Exception as e:
        logger.exception(e)
        raise e
