import os
from ensure import ensure_annotations
from imageSearchEngine.config.configuration import DataIngestionConfig
from imageSearchEngine.logging.logger import log
import zipfile

class DataIngestion:
    '''
    class that would for data ingestion of the required data to the system
    '''
    @ensure_annotations
    def __init__(self, config: DataIngestionConfig):
        '''
        requires: config  and creates the object that would get the data
        '''
        self.config = config
    
    def download_data(self):
        '''
        using the information provided from the config file the method downloads the data to the specificed zip
        returns none
        '''
        path = self.config.local_data_file
        dir = self.config.root_dir
        if not os.path.exists(path):
            os.system(self.config.command + f' -p {dir}')
            log.info(f'successfully downloaded the data from kaggle at {path}')
        else:
            log.info('file already existed download not needed')
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        log.info('done unziping the data')