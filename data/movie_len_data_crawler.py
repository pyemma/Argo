from typing import List
import os

import ray
from imdb import Cinemagoer
import pandas as pd

from data.dataset_manager import dataset_manager, DatasetType


ATTRIBUTION_EXTRACTOR = {
    "imdb_id": lambda m: m.movieID,
    "title": lambda m: m.get("title"),
    "year": lambda m: m.get("year"),
    "votes": lambda m: m.get("votes"),
    "plot_outline": lambda m: m.get("plot outline"),
    "plot_summary": lambda m: (m.get('plot') or [''])[0] if isinstance(m.get('plot'), list) and m.get('plot') else m.get('plot outline') or '',
    "director": lambda m: [c.get("name", "") for c in m.get("director", [])],
    "cast": lambda m: [c.get("name", "") for c in m.get("cast", [])],
    "cover_url": lambda m: m.get("cover url"),
    "kind": lambda m: m.get("kind"),
}

@ray.remote
class MovieLenDataCrawler:
    """
        Define this as a ray actor to crawl the movie data from IMDb. We use ray
        as a distributed computing framework given its popularity in the industry.

        Note that there is no failure recovery mechanism in the current implementation.
    """
    def __init__(self, movie_ids: List[str]):
        self.movie_ids = movie_ids
        self.ia = Cinemagoer()

    def fetch_movie_data(self):
        movie_data = []
        for movie_id in self.movie_ids:
            try:
                movie = self.ia.get_movie(movie_id)
                movie_data.append({
                    key: value(movie) for key, value in ATTRIBUTION_EXTRACTOR.items()
                })
            except Exception as e:
                print(f"Error fetching movie data for {movie_id}: {e}")
                continue
        return pd.DataFrame(movie_data)

if __name__ == "__main__":
    dataset_path = dataset_manager.get_dataset(DatasetType.MOVIE_LENS_LATEST_FULL)
    links = pd.read_csv(os.path.join(dataset_path, "links.csv"))
    imdb_ids = links["imdbId"].tolist()

    imdb_ids = imdb_ids[:16]

    print(f"Processing {len(imdb_ids)} movies")
    NUM_CPU = 8
    batch_size = len(imdb_ids) // NUM_CPU
    print(f"Batch size: {batch_size}")

    print("Ray initializing...")
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=NUM_CPU)
    
    actors = []
    for batch in range(0, len(imdb_ids), batch_size):
        actors.append(MovieLenDataCrawler.remote(imdb_ids[batch:min(batch+batch_size, len(imdb_ids))]))

    results = ray.get([actor.fetch_movie_data.remote() for actor in actors])
    ray.shutdown()
    
    print(results)

