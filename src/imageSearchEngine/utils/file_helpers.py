import os
import yaml
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from imageSearchEngine.logging.logger import log
from imageSearchEngine.exception import CustomException


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    ''' 
    read yaml file from it's path and returns it's values

    Args:
        path_to_yaml (str): path like input

    Returns:
        ConfigBox: ConfigBox type
    '''
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            log.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except Exception as e:
        raise CustomException(e)


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            log.info(f"created directory at: {path}")
