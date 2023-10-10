from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    command: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataCleaningConfig:
    remove_folder_dir: Path
    remove_train_file_dir: Path
    remove_file_extention: str
    remove_zip_dir: Path

@dataclass(frozen=True)
class FeatureRetrivalConfig:
    root_dir: Path
    data_path: Path
    feature_dir: Path
    image_path_list_dir: Path
    image_labels: Path
    include_top: bool
    pooling: str
    input_shape: List
    target_size: List

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_path: Path
    feature_dir: Path
    image_path_list_dir: Path
    n_jobs: int
    n_neighbors: int