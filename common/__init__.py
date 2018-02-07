from typing import List


class Item:
    def __init__(self, item_id: str, name: str, genres: List[str]):
        self.item_id = item_id
        self.name = name
        self.genres = genres

    def __str__(self):
        return "Movie #%s: %s (genres: %s)" % (self.item_id, self.name, ",".join(self.genres))


class ItemRating:
    def __init__(self, user_id: str, item_id: str, rating: float, timestamp: int=0):
        self.user_id = user_id
        self.item_id = item_id
        self.rating = rating
        self.timestamp = timestamp

    def __str__(self):
        return "User %s rated %s (%d)" % (self.user_id, self.item_id, self.rating)


def avg_update(old, count, next):
    return old * (float(count) - 1) / float(count) + next / float(count)