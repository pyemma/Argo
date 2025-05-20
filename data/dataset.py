import numpy as np
import os
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from data.dataset_manager import DatasetType, dataset_manager
from feature.movie_len_features import MovieLenFeatureStore
from feature.utils import padding


class MovieLenDataset(Dataset):

    @classmethod
    def prepare_data(cls, history_seq_length: int = 6, reindex=False):
        """
        This is the core function to process raw MovieLen dataset to generate the
        training samples, including feature engineering, label generation, movie id reindex, etc

        Return the prepare training data, as well as the movie id index mapping
        """
        cached_path = dataset_manager.get_dataset(DatasetType.MOVIE_LENS_LATEST_SMALL)

        # TODO optimization to preprocess data and load on-demand
        links = pd.read_csv(os.path.join(cached_path, "links.csv"))
        movies = pd.read_csv(os.path.join(cached_path, "movies.csv"))
        ratings = pd.read_csv(os.path.join(cached_path, "ratings.csv"))
        links = pd.read_csv(os.path.join(cached_path, "links.csv"))

        unique_ids = None
        if reindex:
            # reindex movie id to compact version
            # use factorize to compute new mapping of ids
            movies["movieId"], unique_ids = pd.factorize(movies["movieId"])
            # use the new mapping of ids as categorical to do assignment, then use code to extract the new ids
            # due to `pd.Categorical`, the movieId is converted to int16, need to convert back to int64
            ratings["movieId"] = pd.Categorical(ratings["movieId"], categories=unique_ids).codes.astype(np.int64)

        # seq feature generation, simplified version, no padding, the minimal
        # length from the ml-latest is 20, for now we assume that
        # sequence feature length + num positive sample <= 20
        user_history_sequence_feature = MovieLenFeatureStore.fetch_item_sequence(ratings=ratings, seq_length=history_seq_length).rename(
            columns={"item_sequence": "history_sequence_feature", "userId": "user_id"}
        )  # user_id, history_sequence_feature

        # positive candidate generation, use rating >= 5.0 as positive
        # negative candidate generation, use rating <= 2.0 as negative
        df = (
            ratings.groupby("userId")
            .apply(lambda x: x.sort_values("timestamp").iloc[history_seq_length:])  # use sequence that is not part of the sequence feature to prevent label leakage
            .reset_index(drop=True)
        )

        positive = (
            df[df["rating"] >= 5.0]
            .groupby("userId")
            .apply(lambda x: x.sort_values("timestamp").head(10))
            .reset_index(drop=True)
            .drop(columns=["rating", "timestamp"])
            .rename(columns={"userId": "user_id", "movieId": "movie_id"})
        )  # user_id, movie_id
        positive["label"] = 1.0

        negative = (
            df[df["rating"] <= 2.0]
            .groupby("userId")
            .apply(lambda x: x.sort_values("timestamp").head(50))
            .reset_index(drop=True)
            .drop(columns=["rating", "timestamp"])
            .rename(columns={"userId": "user_id", "movieId": "movie_id"})
        )
        negative["label"] = 0.0

        training_data = (
            (pd.concat([positive, negative], axis=0).merge(user_history_sequence_feature, on="user_id")).sample(frac=1).reset_index(drop=True)  # user_id, movie_id, label
        )  # shuffle the rows

        return training_data, unique_ids

    def __init__(self, history_seq_length, data, unique_ids):
        self.history_seq_length = history_seq_length
        self.data = data
        self.unique_ids = unique_ids

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        feature = {
            "user_id": torch.tensor([self.data.loc[index]["user_id"]]),
            "item_id": torch.tensor([self.data.loc[index]["movie_id"]]),
            "user_history_behavior": torch.tensor(self.data.loc[index]["history_sequence_feature"]),
            "user_history_length": torch.tensor([self.history_seq_length]),
            "dense_features": torch.ones([8]),
        }
        return feature, torch.tensor(self.data.loc[index]["label"], dtype=torch.float32)


def prepare_movie_len_dataset(history_seq_length: int = 6, eval_ratio: float = 0.2, reindex=False):
    """
    A helper function to prepare movie len dataset, since the MovieLen dataset is not split
    before hand into train/val/eval, I use this function to read the entire dataset and do
    the split of data to avoid potential leakage
    """
    data, unique_ids = MovieLenDataset.prepare_data(history_seq_length=history_seq_length, reindex=reindex)  # this returns all data

    eval_data = data.sample(frac=eval_ratio)
    train_data = data.drop(eval_data.index)

    train_data = train_data.reset_index(drop=True)
    eval_data = eval_data.reset_index(drop=True)

    return (
        MovieLenDataset(
            history_seq_length=history_seq_length,
            data=train_data,
            unique_ids=unique_ids,
        ),
        MovieLenDataset(history_seq_length=history_seq_length, data=eval_data, unique_ids=unique_ids),
        unique_ids,
    )


class MovieLenRatingDataset(Dataset):
    """
    This dataset only load the ratings file. Primarily this is used for
    the MF or the two tower model to compute the item side embeddings.
    """

    def __init__(self, data):
        self.users = torch.LongTensor(data["userId"].values)
        self.movies = torch.LongTensor(data["movieId"].values)
        self.ratings = torch.FloatTensor(data["rating"].values)

    def __len__(self):
        return self.users.shape[0]

    def __getitem__(self, index):
        return self.users[index], self.movies[index], self.ratings[index]


def prepare_movie_len_rating_dataset(eval_ratio=0.1, dataset_type=DatasetType.MOVIE_LENS_LATEST_FULL):
    dataset_path = dataset_manager.get_dataset(dataset_type)
    ratings = pd.read_csv(os.path.join(dataset_path, "ratings.csv"))

    eval_data = ratings.sample(frac=eval_ratio)
    train_data = ratings.drop(index=eval_data.index)

    train_data = train_data.reset_index(drop=True)
    eval_data = eval_data.reset_index(drop=True)

    return MovieLenRatingDataset(train_data), MovieLenRatingDataset(eval_data)


class MovieLenTransActDataset(Dataset):
    """
    This dataset is used for TransAct model, which requires item embeddings to be part
    of the input; right now we have generated the embeddings and saved in file, we need
    to join the embeddings with the sequence read from the ratings.csv file
    """

    # TODO: consider refactor this out as feature function to be reused
    @classmethod
    def prepare_data(cls, history_seq_length: int = 6, max_num_genres: int = 5, dataset_type: DatasetType = DatasetType.MOVIE_LENS_LATEST_SMALL):
        """
        This is the core logic of dataset preparation, it group user's sequence based on
        the chronologic order, and returns the movie id and user ratings which is converted
        to categorize as action

        For now, we only generate the sequence, the last item in the sequence could be used as
        the label for training
        """
        cached_path = dataset_manager.get_dataset(dataset_type)

        ratings = pd.read_csv(os.path.join(cached_path, "ratings.csv"))
        movies = pd.read_csv(os.path.join(cached_path, "movies.csv"))

        feature_store = MovieLenFeatureStore(ratings=ratings, movies=movies, seq_length=history_seq_length)

        # compute the positive/negative label
        neg_idx, pos_idx = set(), set()
        for idx, i in enumerate(feature_store.unique_ids):
            if i < 4.0:
                neg_idx.add(idx)
            else:
                pos_idx.add(idx)

        # obtain item sequence, use the [0:history_seq_length-1] sequence as the input and the last item as
        # candidate and action as label
        # TODO: find more efficient approach instead of making a whole copy

        # when use larger movie len dataset, the dynamic of user increase and it would be more common to encounter
        # user with shorter history, improve the logic here to properly handle the case
        #
        # there are 2 options:
        #   1. remove the users the does not have sufficient history sequence
        #   2. padding, but we need to shift movie ids and the current version of movie embedding would not be useable

        # use option 1 for now
        user_item_sequence = feature_store.item_sequence.copy()  # userId, item_sequence
        user_item_sequence["item_sequence"] = user_item_sequence["item_sequence"].apply(lambda x: x[: history_seq_length - 1])
        user_action_sequence = feature_store.action_sequence.copy()  # userId, action_sequence
        user_action_sequence["action_sequence"] = user_action_sequence["action_sequence"].apply(lambda x: x[: history_seq_length - 1])

        candidate = feature_store.item_sequence.copy()
        candidate["item_sequence"] = candidate["item_sequence"].apply(lambda x: x[history_seq_length - 1])
        candidate = candidate.rename(columns={"item_sequence": "candidate_item"})
        label = feature_store.action_sequence.copy()
        label["action_sequence"] = label["action_sequence"].apply(lambda x: x[history_seq_length - 1])
        label = label.rename(columns={"action_sequence": "label"})

        def labeling(x, neg_idx, pos_idx):
            """
            Simple labeling logic
            """
            if x in neg_idx:
                return 0.0
            if x in pos_idx:
                return 1.0
            raise ValueError(f"Missing index {x}")

        label["label"] = label["label"].apply(lambda x: labeling(x, neg_idx, pos_idx))

        # for genre feature, it is essentially a var-length feature but for input we need to have fix length as of now,
        # here we padding 0 for list
        # TODO, support var-length sparse ids
        user_viewed_genres = feature_store.user_viewed_genres[["userId", "sorted_genres"]].copy()
        user_viewed_genres["sorted_genres"] = user_viewed_genres["sorted_genres"].apply(lambda x: padding(x, max_num_genres))

        # retrieve *document* side feature
        candidate = candidate.merge(feature_store.movie_genres, how="left", left_on="candidate_item", right_on="movieId")
        candidate["genres"] = candidate["genres"].apply(lambda x: padding(x, max_num_genres))

        # merge all features together as a single dataframe
        df = (
            user_item_sequence.merge(user_action_sequence, on="userId")
            .merge(candidate, on="userId")
            .merge(label, on="userId")
            .merge(feature_store.user_num_viewed_movies, on="userId")
            .merge(user_viewed_genres, on="userId")
        )

        return df, feature_store.unique_ids

    def __init__(self, embedding_store, data):
        """
        The item embedding is passed as an object for the dataset to extract the embeddings,
        this is a type of `lazy compute` pattern; for simplicity, we would directly use
        `torch.Tensor` here and in the future this could be generalized to other object
        """
        super().__init__()
        self.embedding_store = embedding_store
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        user_id = self.data.iloc[index]["userId"]
        item_sequence = self.data.iloc[index]["item_sequence"]
        action_sequence = self.data.iloc[index]["action_sequence"]
        candidate_item = self.data.iloc[index]["candidate_item"]
        user_viewed_genres = self.data.iloc[index]["sorted_genres"]
        user_num_viewed_movies = self.data.iloc[index]["num_viewed_movies"]
        candidate_genres = self.data.iloc[index]["genres"]
        # TODO: refactor this to make it more consistent
        return {
            "user_id": torch.LongTensor([user_id]),  # 1,
            "action_sequence": torch.LongTensor(action_sequence),  # seq,
            "item_sequence": self.embedding_store[item_sequence],  # seq x d_item
            "candidate": self.embedding_store[candidate_item],  # d_item
            "user_viewed_genres": torch.LongTensor(user_viewed_genres),  # num_genres
            "user_num_viewed_movies": torch.tensor([user_num_viewed_movies]),  # 1
            "candidate_genres": torch.LongTensor(candidate_genres),  # num_genres
        }, torch.tensor(self.data.iloc[index]["label"], dtype=torch.float)


def prepare_movie_len_transact_dataset(embedding_store, dataset_type, history_seq_length, eval_ratio: float = 0.1):
    data, unique_ids = MovieLenTransActDataset.prepare_data(history_seq_length=history_seq_length, dataset_type=dataset_type)

    eval_data = data.sample(frac=eval_ratio)
    train_data = data.drop(index=eval_data.index)

    train_data = train_data.reset_index(drop=True)
    eval_data = eval_data.reset_index(drop=True)

    return MovieLenTransActDataset(embedding_store, train_data), MovieLenTransActDataset(embedding_store, eval_data)


if __name__ == "__main__":
    # train_dataset, eval_dataset, movie_index = prepare_movie_len_dataset(history_seq_length=8, reindex=True)

    # loader = DataLoader(dataset=train_dataset, batch_size=5)

    # it = iter(loader)
    # batch = next(it)

    # assert batch[0]['user_id'].shape == (5, 1)
    # assert batch[0]['item_id'].shape == (5, 1)
    # assert batch[0]['user_history_behavior'].shape == (5, 8)

    # assert batch[1].shape == (5,)

    # # check type of the output
    # assert batch[0]['user_id'].dtype == torch.int64
    # assert batch[0]['item_id'].dtype == torch.int64

    # print("MovieLenDataset batch shape test passed...")

    # train_dataset, eval_dataset = prepare_movie_len_rating_dataset()

    # loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    # it = iter(loader)
    # batch = next(it)

    # assert len(batch) == 3
    # users, movies, ratings = batch
    # assert len(users) == 5
    # assert len(movies) == 5
    # assert len(ratings) == 5

    # print("MovieLenRatingDataset batch shape test passed...")

    embedding_store = torch.ones((300000, 64))
    train_dataset, eval_dataset = prepare_movie_len_transact_dataset(embedding_store=embedding_store, dataset_type=DatasetType.MOVIE_LENS_LATEST_SMALL, history_seq_length=15)

    loader = DataLoader(train_dataset, batch_size=5)
    it = iter(loader)
    batch, label = next(it)

    assert batch["user_id"].shape == (5, 1)
    assert batch["action_sequence"].shape == (5, 14)
    assert batch["item_sequence"].shape == (5, 14, 64)
    assert batch["candidate"].shape == (5, 64)
    assert batch["user_viewed_genres"].shape == (5, 5)
    assert batch["user_num_viewed_movies"].shape == (5, 1)
    assert batch["candidate_genres"].shape == (5, 5)

    print("MovieLenTransActDataset batch shape test passed...")
