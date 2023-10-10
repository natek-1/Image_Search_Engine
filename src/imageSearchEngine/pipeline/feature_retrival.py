from imageSearchEngine.config.configuration import ConfigurationManager
from imageSearchEngine.components.feature_retrival import FeatureRetrival
from imageSearchEngine.logging.logger import log
from imageSearchEngine.exception import  CustomException

class FeatureRetrivalTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            feature_retrival_config = config.get_feature_retrival_config()
            feature_retrival = FeatureRetrival(config=feature_retrival_config)
            feature_retrival.retrive_labels()
            #feature_retrival.retrive_embedding()
        except Exception as e:
            raise CustomException(e)


