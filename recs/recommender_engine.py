from collections import namedtuple
from typing import List, Dict


class ItemRating:
    def __init__(self, user_id: str, item_id: str, rating: float, timestamp: int=0):
        self.user_id = user_id
        self.item_id = item_id
        self.rating = rating
        self.timestamp = timestamp

    def __str__(self):
        return "User %s rated %s (%d)" % (self.user_id, self.item_id, self.rating)


class RecommenderEngine:
    def __init__(self):
        # User and item lookups. Each entry is guaranteed to have at least 1 record
        self.user_histories = dict()  # type: Dict[str, List[ItemRating]]
        self.item_histories = dict()  # type: Dict[str, List[ItemRating]]

    def add_data(self, user_id: str, item_id: str, rating: float = 5.0, timestamp:int=0) -> None:
        """
        Indicates that userId is interested in itemId
        """
        if user_id not in self.user_histories:
            self.user_histories[user_id] = list()  # defaultdict has some quirks, and I do not like it
        if item_id not in self.item_histories:
            self.item_histories[item_id] = list()
        rating_record = ItemRating(user_id, item_id, rating, timestamp)
        # Adding same record to lookups for memory consumption reduction
        self.item_histories[item_id].append(rating_record)
        self.user_histories[user_id].append(rating_record)

    def online_update_step(self, user_id: str, item_id: str) -> None:
        """
        RS may use approximate methods to update recommendations in this step.
        This step should be computationally cheap.
        """
        pass

    def predict_interests(self, user_id: str, n: int=5) -> List[str]:
        """
        should return a list of 5 itemIds that may be interesting to this userId
        """
        raise NotImplementedError()

    def predict_rating(self, user_id: str, item_id: str) -> float:
        """
        should return predicted rating
        """
        raise NotImplementedError()

    def build(self):
        """
        This is offline part of RS 
        """
        pass
