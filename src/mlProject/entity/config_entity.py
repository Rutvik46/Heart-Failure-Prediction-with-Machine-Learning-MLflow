from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str 
    local_data_file: Path 
    unzip_dir: Path 

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str 
    unzip_data_dir: Path 
    all_schema: dict

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir:str
    train_data_path:Path
    test_data_path:Path
    model_names:str
    model_param:str
    target_column:str

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir:str
    test_data_path:Path
    model_paths:dict
    all_params:dict
    metrics_file_paths: dict
    target_column:str
    mlflow_url:str