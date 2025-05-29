import os
import pickle

import faiss
import torch
import numpy as np
import pandas as pd

from configs.model import OUTPUT_MODEL_PATH
from data.dataset_manager import DatasetType, dataset_manager

"""
This script is used to build ANN index for movie len dataset, which is
used for candidate generation in the retrieval engine.
"""


def build_index(embeddings, index_type: str = "IVF", nlist=1000):
    """
    Build ANN index for movie len dataset

    Args:
        embeddings: torch.Tensor, shape (num_movies, embedding_dim)
        index_type: str, index type, default is "IVF"
        nlist: int, number of clusters, default is 1000
    """
    d = embeddings.shape[1]

    if index_type == "Flat":
        # exact search index (no approximation)
        index = faiss.IndexFlatL2(d)
    elif index_type == "IVF":
        # inverted file index with clustering
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        # train the index on data
        index.train(embeddings)
    elif index_type == "HNSW":
        # Hierarchical Navigable Small World Graph
        index = faiss.IndexHNSWFlat(d, M=16)
    else:
        raise ValueError(f"Invalid index type: {index_type}")

    # add embeddings to index
    index.add(embeddings)
    return index


def save_index(index, movie_id_remapper):
    faiss.write_index(index, os.path.join(OUTPUT_MODEL_PATH, "movies.index"))
    with open(os.path.join(OUTPUT_MODEL_PATH, "movie_id_remapper.pkl"), "wb") as file:
        pickle.dump(movie_id_remapper, file)


if __name__ == "__main__":
    embeddings = torch.load(os.path.join(OUTPUT_MODEL_PATH, "movie_embeddings_v2.pt"), weights_only=False)
    # here we need to do a normalization to the embeddings because we use a hacky solution when generate
    # the movie embeddings without reindex, thus ending with lots of phantom embeddings. They would pollute
    # our index, and thus we would remove them before the index building
    dataset_path = dataset_manager.get_dataset(DatasetType.MOVIE_LENS_LATEST_FULL)
    movies = pd.read_csv(os.path.join(dataset_path, "movies.csv"))
    movie_ids = movies["movieId"].tolist()
    movie_id_remapper = {idx: movie_id for idx, movie_id in enumerate(movie_ids)}
    embeddings = embeddings[movie_ids]
    print(embeddings.shape)

    index = build_index(embeddings)

    # A simple verification on the EBR result
    # FAISS expects a 2D tensor input, B x d
    query_embedding = np.expand_dims(embeddings[0], axis=0)

    distances, indices = index.search(query_embedding, 4)
    movies = movies.set_index("movieId")
    for idx in indices[0]:
        movie_id = movie_id_remapper[idx]
        print(movies.loc[movie_id])

    save_index(index, movie_id_remapper)
