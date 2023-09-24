from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier import logger
from cnn_classifier.components.evaluation import Evaluation

STAGE_NAME = "Model Evaluation"

class  ModelEvaluationPipeline(object):
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_validation_config()
        eval = Evaluation(config=eval_config)
        eval.evaluation()

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
