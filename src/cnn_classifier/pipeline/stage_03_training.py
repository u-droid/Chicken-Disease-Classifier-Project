from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier import logger
from cnn_classifier.components.training import Training

STAGE_NAME = "Model Training"

class  ModelTrainingPipeline(object):
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        train_loader, valid_loader = training.get_data_loaders()
        training.train(
            train_loader=train_loader,
            valid_loader=valid_loader
        )

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
