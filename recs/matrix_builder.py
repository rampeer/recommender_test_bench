from typing import Dict, List

from scipy.sparse import dok_matrix
import numpy as np

from recs.recommender_engine import ItemRating


class MatrixBuilder:
    def __init__(self, user_histories: Dict[str, List[ItemRating]], item_histories: Dict[str, List[ItemRating]]):
        self.m = dok_matrix((len(user_histories), len(item_histories)))
        self.item_avg_rating = {x: np.mean([r.rating for r in y]) for x, y in item_histories.items()}  # type: Dict[str, float]
        self.user_avg_rating = {x: np.mean([r.rating for r in y]) for x, y in user_histories.items()}  # type: Dict[str, float]
        self.user_row_index = dict()  # type: Dict[str, int]
        self.item_col_index = dict()  # type: Dict[str, int]
        for col_index, (item_id, histories) in enumerate(item_histories.items()):
            self.item_col_index[item_id] = col_index
            for record in histories:
                row_index = self.user_row_index.get(record.user_id, len(self.user_row_index))
                self.user_row_index[record.user_id] = row_index
                self.m[row_index, col_index] = record.rating

    def get_rating(self, user_id: str, item_id: str) -> float:
        return self.m[self.user_row_index[user_id], self.item_col_index[item_id]]
