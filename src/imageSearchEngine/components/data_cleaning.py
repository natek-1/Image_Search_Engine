import os
import shutil
from imageSearchEngine.logging.logger import log
from imageSearchEngine.exception import CustomException
from imageSearchEngine.config.configuration import DataCleaningConfig
from ensure import ensure_annotations

class DataCleaning:
    '''class that would take care of cleaning the data in current directories'''

    @ensure_annotations
    def __init__(self, config: DataCleaningConfig):
        self.config = config
    
    def clean(self):
        '''
        Cleans data as per request should only be done once
        '''
        try:
            
            shutil.rmtree(self.config.remove_folder_dir)
            log.info(f'Removed {self.config.remove_folder_dir}')
            os.remove(self.config.remove_zip_dir)
            log.info(f'Removed {self.config.remove_zip_dir}')
            for folder in os.listdir(self.config.remove_train_file_dir):
                current_dir = os.path.join(self.config.remove_train_file_dir, folder)
                for items in os.listdir(current_dir):
                    if items.endswith(f'.{self.config.remove_file_extention}'):
                        os.remove(os.path.join(current_dir, items))
            log.info(f'remove all the files ending with {self.config.remove_file_extention} on the directory {self.config.remove_train_file_dir}')
        except Exception as e:
            raise CustomException(e)

        