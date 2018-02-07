import pandas as pd
from typing import Iterable, List, Dict
from .loader import Loader
from common import Item, ItemRating


class MovieLensLoader(Loader):
    def __init__(self, path: str, clip: int = 100000):
        self.ratings = pd.read_csv(path + "/ratings.csv", nrows=clip, dtype={
            "userId": str,
            "movieId": str,
            "rating": float
        })

        movies = pd.read_csv(path + "/movies.csv", dtype={
            "movieId": str,
            "title": str,
            "genres": str
        })

        self.movies = {item_id: Item(item_id, name, genres.split("|"))
                       for item_id, name, genres in movies[["movieId", "title", "genres"]].values}  # type: Dict[str, Item]

    def get_records(self):
        for user_id, item_id, rating, timestamp in self.ratings[["userId", "movieId", "rating", "timestamp"]].values:
            yield ItemRating(user_id, item_id, rating, timestamp)

    def get_item_description(self, item_id):
        return str(self.movies.get(item_id, "Unknown"))

    def get_item_genres(self, item_id: str):
        if item_id in self.movies:
            return self.movies[item_id].genres
        else:
            return []
