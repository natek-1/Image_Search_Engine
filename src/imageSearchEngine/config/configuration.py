from ensure import ensure_annotations
from imageSearchEngine.constants import *
from imageSearchEngine.utils.file_helpers import read_yaml, create_directories
from imageSearchEngine.entity import DataIngestionConfig, DataCleaningConfig

class ConfigurationManager:
    @ensure_annotations
    def __init__(
        self,
        config_filepath:Path = CONFIG_FILE_PATH,
        params_filepath:Path = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        #self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    @ensure_annotations
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        '''
        Creates and returns the configuration for data ingestion
        the object contains the attritubes needed for the data ingestion portion of our project
        '''

        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            command=config.command,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )
        return data_ingestion_config

    @ensure_annotations
    def get_clean_data_config(self) -> DataCleaningConfig:
        '''
        Creates and returns the configuration for data cleaning
        the object contains the attritubes needed for the data cleaning portion of our project
        '''
        config = self.config.data_cleaning

        data_cleaning_config = DataCleaningConfig(
            remove_folder_dir=config.remove_folder_dir,
            remove_train_file_dir=config.remove_train_file_dir,
            remove_file_extention=config.remove_file_extention,
            remove_zip_dir=config.remove_zip_dir
        )
        return data_cleaning_config