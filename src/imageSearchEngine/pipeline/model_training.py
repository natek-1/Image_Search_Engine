from imageSearchEngine.config.configuration import ConfigurationManager
from imageSearchEngine.components.model_trainer import ModelTrainer
from imageSearchEngine.logging.logger import log
from imageSearchEngine.exception import  CustomException


class ModelTrainingTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer = ModelTrainer(config= model_trainer_config)
            model_trainer.train_model()
        except Exception as e:
            raise CustomException(e)

