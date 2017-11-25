from collections import Counter

from recs import RecommenderEngine
from scipy.sparse import csc_matrix, dok_matrix
from collections import Counter


class ItemCF(RecommenderEngine):
    def __init__(self, consider_top_users = 1000):
        super().__init__()
        self.consider_top_users = consider_top_users

    def build(self):
        user_history_lens = Counter({u: len(hist) for u, hist in self.user_histories.items()})

        user_row_index = dict()  # type: Dict[str, int]
        item_col_index = dict()  # type: Dict[str, int]

        for user_id, _ in user_history_lens.most_common(self.consider_top_users):
            for record in self.user_histories.get(user_id, []):
                item_col_index[record.item_id] = col_index
                avg_rating = np.mean([x.rating for x in histories])
                self.item_average_rating[item_id] = avg_rating
                for record in histories:
                    row_index = user_row_index.get(record.user_id, len(user_row_index))
                    user_row_index[record.user_id] = row_index
                    m[row_index, col_index] = record.rating - avg_rating
        self.global_average = np.mean(list(self.item_average_rating.values()))