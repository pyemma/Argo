from dataclasses import dataclass
from enum import Enum, auto
import os
import logging
import requests
import zipfile
import io


@dataclass
class Dataset:
    path: str


class DatasetType(Enum):
    MOVIE_LENS_LATEST_SMALL = auto()  # 100k ratings, 3600 tags, 9000 movies, 600 users, updated 9/2018

    def get_url(self) -> str:
        urls = {
            DatasetType.MOVIE_LENS_LATEST_SMALL: "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
        }
        return urls[self]
    
    def get_name(self) -> str:
        names = {
            DatasetType.MOVIE_LENS_LATEST_SMALL: "movie_lens_latest_small",
        }
        return names[self]


class DatasetManager:
    """
        This is a manager class to handle the dataset on the host, the dataset would
        be downloaded and cached in the /tmp folder by default
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DatasetManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, tmp_folder: str = "/tmp"):
        if self._initialized:
            return
        
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)

        self._tmp_folder = tmp_folder
        self._initialized = True

    def get_dataset(self, dataset_type: DatasetType) -> Dataset:
        dataset_path = os.path.join(self._tmp_folder, dataset_type.get_name())
        if not os.path.exists(dataset_path):
            logging.info(f"Requested dataset {dataset_type} does not exists in cache, start downloading")
            self.download_dataset(dataset_type=dataset_type)

        return Dataset(path=dataset_path)

    def download_dataset(self, dataset_type: DatasetType) -> str:
        logging.info(f"Start downloading {dataset_type}...")
        dataset_path = os.path.join(self._tmp_folder, dataset_type.get_name())
        if os.path.exists(dataset_path):
            return
        
        # TODO implement the dataset downloading logic here

if __name__ == "__main__":
    dataset_manager = DatasetManager()

    dataset_manager.get_dataset(DatasetType.MOVIE_LENS_LATEST_SMALL)