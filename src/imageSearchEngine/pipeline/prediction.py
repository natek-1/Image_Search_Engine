from imageSearchEngine.config.configuration import ConfigurationManager
from imageSearchEngine.components.prediction import Prediction
from imageSearchEngine.exception import  CustomException


try:
    configuration = ConfigurationManager()
    model_evaluation_config =  configuration.get_prediction_config()
    model_evaluation = Prediction(config=model_evaluation_config)
    print(model_evaluation.predict("artifacts/data_ingestion/tiny-imagenet-200/val/n01629819/images/val_3592.JPEG"))
except Exception as e:
    raise CustomException(e)