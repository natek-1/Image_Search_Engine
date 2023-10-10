from ensure import ensure_annotations
from imageSearchEngine.constants import *
from imageSearchEngine.utils.file_helpers import read_yaml, create_directories
from imageSearchEngine.entity import DataIngestionConfig, DataCleaningConfig, FeatureRetrivalConfig, ModelTrainerConfig, ModelEvaluationConfig

class ConfigurationManager:
    @ensure_annotations
    def __init__(
        self,
        config_filepath:Path = CONFIG_FILE_PATH,
        params_filepath:Path = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

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
    

    @ensure_annotations
    def get_feature_retrival_config(self) -> FeatureRetrivalConfig:
        '''
        Creates and returns the configuration to make the feature representation
        the object contains the attritubes needed for the retrive those features from the images portion of our project
        '''
        config = self.config.feature_representation
        params = self.params.feature_representation
        create_directories([config.root_dir])

        feature_retrival_config = FeatureRetrivalConfig(
            root_dir = config.root_dir,
            data_path = config.data_path,
            feature_dir = config.feature_dir,
            image_path_list_dir= config.image_path_list_dir,
            image_labels= config.image_labels,
            include_top= params.include_top,
            pooling = params.pooling,
            input_shape= params.input_shape,
            target_size= params.target_size

        )
        return feature_retrival_config
    
    @ensure_annotations
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        '''
        Creates and returns the configuration to train the model
        the object contains the attritubes needed properly train the mode
        '''
        config = self.config.model_trainer
        params = self.params.model_trainer
        create_directories([config.root_dir])
        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            model_path=config.model_path,
            n_jobs=params.n_jobs,
            n_neighbors=params.n_neighbors,
            feature_dir = config.feature_dir,
            image_path_list_dir= config.image_path_list_dir

        )

        return model_trainer_config

    @ensure_annotations
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        '''
        Creates and returns the configuration to evaluate the model on the test dataset
        the object contains the attritubes needed properly see the model score
        '''
        config = self.config.model_evaluation
        params = self.params.model_evaluation
        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            model_path= config.model_path,
            image_labels_path= config.image_labels_path,
            image_path_list_dir = config.image_path_list_dir,
            val_feature= config.val_feature,
            val_path=config.val_path,
            n_neighbors=params.n_neighbors,
            return_distance= params.return_distance,
            include_top= params.include_top,
            pooling = params.pooling,
            input_shape= params.input_shape,
            target_size= params.target_size
        )
        return model_evaluation_config
