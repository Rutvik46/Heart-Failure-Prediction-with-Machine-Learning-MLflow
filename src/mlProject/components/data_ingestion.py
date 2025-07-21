from urllib import request
import zipfile
from mlProject.utils.common import get_size
from mlProject import logger
import os
from pathlib import Path
from mlProject.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):

        if not os.path.exists(self.config.local_data_file):
            file_name, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )

            logger.info(f"{file_name} downloaded with following info: {headers}")
            logger.info(f"File size: {Path(get_size(self.config.local_data_file))}")

        else:
            logger.info(f"File {self.config.local_data_file} already exists.")

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