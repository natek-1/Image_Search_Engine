from dataclasses import dataclass
from pathlib import Path

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