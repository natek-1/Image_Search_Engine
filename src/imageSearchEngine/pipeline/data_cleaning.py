from imageSearchEngine.config.configuration import ConfigurationManager
from imageSearchEngine.components.data_cleaning import DataCleaning
from imageSearchEngine.logging.logger import log
from imageSearchEngine.exception import  CustomException

class DataCleanTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_clean_data_config()
            data_cleaning = DataCleaning(config=data_ingestion_config)
            data_cleaning.clean()
        except Exception as e:
            raise CustomException(e)

