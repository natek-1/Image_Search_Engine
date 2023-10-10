from imageSearchEngine.pipeline.data_ingestion import DataIngestionTrainingPipeline
from imageSearchEngine.pipeline.data_cleaning import DataCleanTrainingPipeline
from imageSearchEngine.pipeline.feature_retrival import FeatureRetrivalTrainingPipeline
from imageSearchEngine.logging.logger import log
from imageSearchEngine.exception import CustomException

try:
    stage_name = "Data Ingestion stage"
    log.info(f"\n\n{'>'*10} {stage_name}  started {'<'*10}")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    log.info(f"{'>'*10} {stage_name} completed {'<'*10} \n\n x{'='*20}x \n\n")
    stage_name = "Data Cleaning stage"
    log.info(f"\n\n{'>'*10} {stage_name}  started {'<'*10}")
    data_cleaning = DataCleanTrainingPipeline()
    data_cleaning.main()
    log.info(f"{'>'*10} {stage_name} completed {'<'*10} \n\n x{'='*20}x \n\n")
    stage_name = "Feature Retrival stage"
    log.info(f"\n\n{'>'*10} {stage_name}  started {'<'*10}")
    feature_retrival = FeatureRetrivalTrainingPipeline()
    feature_retrival.main()
    log.info(f"{'>'*10} {stage_name} completed {'<'*10} \n\n x{'='*20}x \n\n")
except Exception as e:
    raise CustomException(e)
