from typing import Counter

from recs import RecommenderEngine
from common import avg_update
from collections import Counter


class AverageRatingRecs(RecommenderEngine):
    def __init__(self):
        super().__init__()
        self.item_rating_count = Counter()  # type: Counter[str, int]
        self.item_average_rating = Counter()  # type: Counter[str, float]
        self.global_rating_count = 0
        self.global_average = 2.5

    def add_data(self, user_id: str, item_id: str, rating: float = 5.0, timestamp: int = 0):
        if item_id not in self.item_rating_count:
            self.item_average_rating[item_id] = rating
            self.item_rating_count[item_id] = 1
        else:
            self.item_rating_count[item_id] += 1
            current_count = float(self.item_rating_count[item_id])
            # Average update formula
            self.item_average_rating[item_id] = avg_update(self.item_average_rating[item_id], current_count, rating)
        self.global_rating_count += 1
        self.global_average = avg_update(self.global_average, self.global_rating_count, rating)
        super().add_data(user_id, item_id, rating)

    def predict_rating(self, user_id: str, item_id: str):
        return self.item_average_rating.get(item_id, self.global_average)

    def predict_interests(self, user_id: str, n: int = 5):
        return [x for x, _ in self.item_average_rating.most_common(n)]
