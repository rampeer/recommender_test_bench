from typing import Iterable

from common import ItemRating


class Loader:
    """
    It is useful to put data loading/mangling in one separate class.
    This way we can test our RSs on different datasets without changing their logic (in theory at least)
    """

    def get_records(self) -> Iterable[ItemRating]:
        raise NotImplementedError()

    def get_item_genres(self, item_id):
        return []

    def get_item_description(self, item_id) -> str:
        raise NotImplementedError()

