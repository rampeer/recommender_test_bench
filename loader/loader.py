from typing import Iterable, Tuple

from common import ItemRating, Item


class Loader:
    """
    It is useful to put data loading/mangling in one separate class.
    This way we can test our RSs on different datasets without changing their logic (in theory at least)
    """

    def get_records(self) -> Iterable[ItemRating]:
        raise NotImplementedError()

    def get_items(self) -> Iterable[Tuple[str, Item]]:
        raise NotImplementedError()

    def put_record(self, item_id: str, user_id: str, rating: float, timestamp: int):
        pass

    def get_item_genres(self, item_id):
        return []

    def get_item_description(self, item_id) -> str:
        raise NotImplementedError()

