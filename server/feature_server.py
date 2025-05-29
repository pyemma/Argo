from dataclasses import dataclass
from enum import Enum, auto
import os

import numpy as np
import pandas as pd

import torch

from data.dataset_manager import DatasetType, dataset_manager
from feature.movie_len_features import MovieLenFeatureStore
from feature.utils import padding


@dataclass
class FeatureServerConfig:
    movie_len_history_seq_length: int
    movie_len_dataset_type: DatasetType
    embedding_store_path: str


class FeatureName(Enum):
    # user side features
    ACTION_SEQUENCE = auto()
    ITEM_SEQUENCE = auto()
    ITEM_SEQUENCE_EMBEDDING = auto()
    USER_VIEWED_MOVIE_GENRES = auto()
    NUM_USER_VIEWED_MOVIES = auto()

    # item side features
    MOVIE_GENRES = auto()
    ITEM_EMBEDDING = auto()


class FeatureServer:
    """
    This is a component to handle the feature extraction for a given entity.

    For now we only support the movie len related feature extraction.
    """

    USER_SIDE_FEATURES = [
        FeatureName.ACTION_SEQUENCE,
        FeatureName.ITEM_SEQUENCE,
        FeatureName.NUM_USER_VIEWED_MOVIES,
        FeatureName.USER_VIEWED_MOVIE_GENRES,
    ]

    ITEM_SIDE_FEATURES = [
        FeatureName.MOVIE_GENRES,
        # FeatureName.ITEM_EMBEDDING, this is also item side feature but we would process it individually
    ]

    def __init__(self, config: FeatureServerConfig):
        self.store = {}

        # we use a precompute strategy to populate the feature values and hold them in memory as of now
        cached_path = dataset_manager.get_dataset(config.movie_len_dataset_type)

        movies = pd.read_csv(os.path.join(cached_path, "movies.csv"))
        ratings = pd.read_csv(os.path.join(cached_path, "ratings.csv"))

        feature_store = MovieLenFeatureStore(ratings=ratings, movies=movies, seq_length=config.movie_len_history_seq_length)

        # the original feature data from feature store is guarantee to be the same order and not
        # guarantee to have id and userId exact matched due to the recent introduced filtering logic,
        # here we move the data out and reindex the data by userId and movieId
        self.store[FeatureName.ACTION_SEQUENCE] = feature_store.action_sequence.set_index("userId")
        self.store[FeatureName.ITEM_SEQUENCE] = feature_store.item_sequence.set_index("userId")
        self.store[FeatureName.MOVIE_GENRES] = feature_store.movie_genres.set_index("movieId")
        self.store[FeatureName.MOVIE_GENRES]["genres"] = self.store[FeatureName.MOVIE_GENRES]["genres"].apply(lambda x: padding(x, 3))
        self.store[FeatureName.USER_VIEWED_MOVIE_GENRES] = feature_store.user_viewed_genres.set_index("userId")
        self.store[FeatureName.USER_VIEWED_MOVIE_GENRES]["sorted_genres"] = self.store[FeatureName.USER_VIEWED_MOVIE_GENRES]["sorted_genres"].apply(lambda x: padding(x, 3))
        self.store[FeatureName.NUM_USER_VIEWED_MOVIES] = feature_store.user_num_viewed_movies.set_index("userId")

        self.embedding_store = torch.load(config.embedding_store_path, weights_only=False)
        self.unique_ids = feature_store.unique_ids

    def generate_feature(self, feature_name, user_id):
        return self.store.get(feature_name).get(user_id)

    def extract_user_feature(self, user_id: int):
        """
        Extract user side feature

        Return feature in feature dict format, indexed by feature name
        """
        # TODO: make this consistent with the offline training data preparation
        # usually in industry, the feature used for offline training is being logged during online inference
        # to make sure the consistency, unless in the early stage of model development where we don't have
        # servable model online yet and we need to do lots of offline injection
        feature_dict = {k: self.store[k].loc[user_id].iloc[0] for k in self.USER_SIDE_FEATURES}

        # process the item sequence and convert to embedding format
        feature_dict[FeatureName.ITEM_SEQUENCE_EMBEDDING] = self.embedding_store[feature_dict[FeatureName.ITEM_SEQUENCE]]
        return feature_dict

    def extract_item_feature(self, item_id: int):
        """
        Extract item side feature

        Return feature in feature dict format, indexed by feature name
        """
        feature_dict = {k: self.store[k].loc[item_id].iloc[0] for k in self.ITEM_SIDE_FEATURES}
        feature_dict[FeatureName.ITEM_EMBEDDING] = self.embedding_store[item_id]
        return feature_dict

    def extract_user_interaction_sequence(self, user_id: int, ratings: float):
        """
        Only extract the user item sequence with rating greater than certain threshold
        """
        item_sequence = self.store[FeatureName.ITEM_SEQUENCE].loc[user_id].iloc[0]
        action_sequence = self.store[FeatureName.ACTION_SEQUENCE].loc[user_id].iloc[0]

        # the action sequence is encoded as categorical feature right now, we need to use the
        # unique ids to check the category we need
        whitelist = set()
        for idx, i in enumerate(self.unique_ids):
            if i >= ratings:
                whitelist.add(idx)

        result = [i for (i, a) in zip(item_sequence, action_sequence) if a in whitelist]
        return result


if __name__ == "__main__":
    server = FeatureServer(
        FeatureServerConfig(movie_len_history_seq_length=15, movie_len_dataset_type=DatasetType.MOVIE_LENS_LATEST_SMALL, embedding_store_path="artifacts/movie_embeddings.pt")
    )

    # user_feature = server.extract_user_feature(user_id=1)
    # item_feature = server.extract_item_feature(item_id=1)

    # print(user_feature)
    # print(item_feature)

    result = server.extract_user_interaction_sequence(user_id=2, ratings=3.0)
    print(result)
