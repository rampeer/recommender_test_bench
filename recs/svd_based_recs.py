import typing

from typing import Dict, List, Optional, Set

from recs import RecommenderEngine
from sparsesvd import sparsesvd
import numpy as np
from scipy.sparse import csc_matrix, dok_matrix
from collections import Counter


class SVDBasedCF(RecommenderEngine):
    def __init__(self, components: int = 40, include_avg_rating: bool = False):
        super().__init__()
        self.components = components
        self.global_average = 2.5
        self.include_avg_rating = include_avg_rating

        self.user_vectors = dict()  # type: Dict[str, np.array]
        self.item_vectors = dict()  # type: Dict[str, np.array]
        self.item_average_rating = dict()  # type: Dict[str, float]

        # Stuff for half-update step
        self.rating_matrix_demeaned = None # type: Optional[dok_matrix]
        self.col_vectors = None  # type: Optional[np.array]
        self.col_vectors_inv = None # type: Optional[np.array]
        self.user_row_index = dict()  # type: Dict[str, int]
        self.item_col_index = dict()  # type: Dict[str, int]

    def online_update_step(self, user_id: str, item_id: str):
        if self.col_vectors_inv is not None:
            # Online update step. This is easily inferred from SVD formula.
            # We cannot update average rating, however, as it invalidates all rows (i.e. we have to recompute SVD).
            rating = self.global_average
            for r in self.user_histories[user_id]:
                if r.item_id == item_id:
                    rating = r.rating
            if item_id in self.item_col_index:
                if user_id in self.user_row_index:
                    row_index = self.user_row_index[user_id]
                else:
                    # This is efficient matrix because we keep demeaned rating in dok-matrix
                    row_index = self.rating_matrix_demeaned.shape[0]
                    self.rating_matrix_demeaned.resize((
                        row_index + 1,
                        self.rating_matrix_demeaned.shape[1]
                    ))
                    self.user_row_index[user_id] = row_index
                self.rating_matrix_demeaned[row_index, self.item_col_index[item_id]] = rating - self.item_average_rating[item_id]
                # what
                self.user_vectors[user_id] = self.rating_matrix_demeaned[row_index].dot(self.col_vectors_inv)[0]

    def build(self):
        # Strictly speaking, enumeration order of Python dict's is not defined.
        # In other words, iterating over same dict twice may yield different results. So, we have to keep mapping
        # between user/item, and row/column.
        self.user_row_index = dict()  # type: Dict[str, int]
        self.item_col_index = dict()  # type: Dict[str, int]
        self.rating_matrix_demeaned = dok_matrix((len(self.user_histories), len(self.item_histories)))

        for col_index, (item_id, histories) in enumerate(self.item_histories.items()):
            self.item_col_index[item_id] = col_index
            avg_rating = np.mean([x.rating for x in histories])
            self.item_average_rating[item_id] = avg_rating
            for record in histories:
                row_index = self.user_row_index.get(record.user_id, len(self.user_row_index))
                self.user_row_index[record.user_id] = row_index
                self.rating_matrix_demeaned[row_index, col_index] = record.rating - avg_rating
        self.global_average = np.mean(list(self.item_average_rating.values()))
        u, s, v = sparsesvd(csc_matrix(self.rating_matrix_demeaned), self.components)

        row_vectors = u.T
        self.col_vectors = np.dot(np.diag(s), v).T
        self.col_vectors_inv = np.linalg.pinv(self.col_vectors.transpose())
        for u, row in self.user_row_index.items():
            self.user_vectors[u] = row_vectors[row]
        for i, col in self.item_col_index.items():
            self.item_vectors[i] = self.col_vectors[col]

    def predict_rating(self, user_id: str, item_id: str):
        pers_rating = 0.0
        if user_id in self.user_vectors and item_id in self.item_vectors:
            pers_rating = np.dot(self.user_vectors[user_id], self.item_vectors[item_id])
        return pers_rating + self.item_average_rating.get(item_id, self.global_average)

    def predict_interests(self, user_id: str, n: int = 5):
        if self.col_vectors is None:
            return []
        affinities = np.dot(self.col_vectors, self.user_vectors.get(user_id, np.zeros(self.components)))
        predicted_ratings = Counter()  # type: typing.Counter[str]
        if self.include_avg_rating:
            for item_id, col in self.item_col_index.items():
                predicted_ratings[item_id] = affinities[col] + self.item_average_rating[item_id]
        else:
            for item_id, col in self.item_col_index.items():
                predicted_ratings[item_id] = affinities[col]
        # Excluding seen items
        for record in self.user_histories.get(user_id, []):
            predicted_ratings.pop(record.item_id, None)
        return [x for x, _ in predicted_ratings.most_common(n)]
