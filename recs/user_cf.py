from typing import Set, Dict
from collections import Counter
from scipy.spatial.distance import cosine

from recs import RecommenderEngine
from recs.matrix_builder import MatrixBuilder
import numpy as np

class UserBasedNNCF(RecommenderEngine):
    """
    This class contains simple user-based nearest neighbour collaborative filtering algorithm.
    The premise of the method is simple: we find nearest neighbours to current user, and use rating of this
    neighbourhood to estimate ratings and make predictions.

    Parameter <neighbour_size> is used to control how many users are included in neighbourhood.

    Usually we use Pearson correlation between user ratings (taking into account only co-rated items). However, we
    might want to subtract user or item averages to remove the bias, as describer in
    {Yao, G. (n.d.). User-Based and Item-Based Collaborative Filtering Recommendation Algorithms Design.}.
    Parameter correction_mode controls this behaviour.

    After calculating neighbourhood we have to estimate ratings. We can do this by calculating weighted average of ratings (see
    {Sarwar, B., Karypis, G., Konstan, J., & Reidl, J. (2001). Item-based collaborative filtering recommendation algorithms. Proceedings of the Tenth International Conference on World Wide Web  - WWW ’01, 285–295. https://doi.org/10.1145/371920.372071}
    ) or, again, we can subtract users' ratings averages before that procedure,  as described in
    {Resnick, P., Iacovou, N., Suchak, M., Bergstrom, P., & Riedl, J. (1994). GroupLens : An Open Architecture for Collaborative Filtering of Netnews. Proceedings of the 1994 ACM Conference on Computer Supported Cooperative Work, 175–186. https://doi.org/10.1145/192844.192905}
    This is controlled by prediction_mode parameter.
    """
    CORRECTION_NONE = "none"
    CORRECTION_USER_MEAN = "user_mean"
    CORRECTION_ITEM_MEAN = "item_mean"

    PREDICTION_AVERAGE = "avg"
    PREDICTION_UNBIASED_AVERAGE = "unbiased_avg"

    def __init__(self, neighbour_size: int = 5,
                 correction_mode: str = CORRECTION_NONE,
                 prediction_mode: str = PREDICTION_AVERAGE,
                 neighbour_sample_max_size: int = 20):
        self.correction_mode = correction_mode
        self.neighbour_size = neighbour_size
        self.prediction_mode = prediction_mode
        self.neighbour_sample_max_size = neighbour_sample_max_size
        self.rating_matrix = None  # type: MatrixBuilder
        super().__init__()

    def build(self):
        self.rating_matrix = MatrixBuilder(self.user_histories, self.item_histories)

        print("Matrix built")
        self._user_lookup = {k: set([x.item_id for x in v]) for k, v in self.user_histories.items()}
        self._item_lookup = {k: set([x.user_id for x in v]) for k, v in self.item_histories.items()}
        print("Lookups built")
        if self.correction_mode == self.CORRECTION_ITEM_MEAN:
            for u, history in self.user_histories.items():
                for r in history:
                    self.rating_matrix.m[self.rating_matrix.user_row_index[u], self.rating_matrix.item_col_index[r.item_id]] -= \
                        self.rating_matrix.item_avg_rating[r.item_id]
        elif self.correction_mode == self.CORRECTION_USER_MEAN:
            for u, history in self.user_histories.items():
                for r in history:
                    self.rating_matrix.m[self.rating_matrix.user_row_index[u], self.rating_matrix.item_col_index[r.item_id]] -= \
                        self.rating_matrix.user_avg_rating[u]
        print("Subtracted bias")


    def _get_similar_user_candidates(self, user_id: str, item_id: str, cap: int) -> Set[str]:
        if item_id not in self._item_lookup:
            return set()
        cands = Counter()
        user_items = self._user_lookup[user_id]
        for u in self._item_lookup[item_id]:
            l = len(self._user_lookup[u] & user_items)
            if l > 0:
                cands[u] = l
        cands.pop(user_id, None)
        return set([x for x, _ in cands.most_common(cap)])

    def _get_similarity(self, user_a: str, user_b: str) -> float:
        return 1.0
        corated_items = self._user_lookup[user_a] & self._user_lookup[user_b]
        corated_items_indices = [self.rating_matrix.item_col_index[x] for x in corated_items]
        v_a = self.rating_matrix.m[self.rating_matrix.user_row_index[user_a], corated_items_indices].todense()
        v_b = self.rating_matrix.m[self.rating_matrix.user_row_index[user_b], corated_items_indices].todense()
        cos = cosine(v_a, v_b)
        if np.isnan(cos):
            return 0.0
        else:
            return cos

    def predict_rating(self, user_id: str, item_id: str):
        cand_sims = Counter()
        cands = self._get_similar_user_candidates(user_id, item_id, self.neighbour_sample_max_size) & self._item_lookup.get(item_id, set())

        for cand in cands:
            cand_sims[cand] = self._get_similarity(user_id, cand)
        if len(cand_sims) == 0:
            return self.rating_matrix.user_avg_rating[user_id]

        best_candidates = dict(cand_sims.most_common(self.neighbour_size))  # type: Dict[str, float]
        if self.prediction_mode == self.PREDICTION_AVERAGE:
            acc = 0.0
            for c, w in best_candidates.items():
                acc += w * (self.rating_matrix.get_rating(c, item_id) - self.rating_matrix.user_avg_rating[c])
            return acc / (sum(np.abs([x for _, x in best_candidates.items()])) + 1e-7) + self.rating_matrix.user_avg_rating[user_id]
        else:
            acc = np.sum([w * self.rating_matrix.get_rating(c, item_id) for c, w in best_candidates.items()])
            return acc / (sum(np.abs([x for _, x in best_candidates.items()])) + 1e-7)
