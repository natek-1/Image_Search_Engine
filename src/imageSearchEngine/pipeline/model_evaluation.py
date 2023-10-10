from imageSearchEngine.config.configuration import ConfigurationManager
from imageSearchEngine.components.model_evaluation import ModelEvaluation
from imageSearchEngine.logging.logger import log
from imageSearchEngine.exception import  CustomException

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            configuration = ConfigurationManager()
            model_evaluation_config =  configuration.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            model_evaluation.evaluate_model()

        except Exception as e:
            raise CustomException(e)