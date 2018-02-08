import pandas as pd
from typing import Iterable, List, Dict, Tuple
from .loader import Loader
from common import Item, ItemRating
import postgresql

class PostgresLoader(Loader):
    def __init__(self, ps_login: str, ps_password: str, ps_host: str, ps_port: int, ps_db: str, init_db: bool = False):
        if init_db:
            with postgresql.open('pq://%s:%s@%s:%s/' % (ps_login, ps_password, ps_host, ps_port)) as db:
                if len(db.query("SELECT * FROM pg_catalog.pg_database WHERE datname=$1", ps_db)) == 0:
                    db.execute("CREATE DATABASE %s" % (ps_db))
        self.connection = postgresql.open('pq://%s:%s@%s:%s/%s' % (ps_login, ps_password, ps_host, ps_port, ps_db))
        if init_db:
            self._init_db()
        self._item_map = None  # type: Dict[str, Item]
        self._insert_items = self.connection.prepare("INSERT INTO ratings VALUES ($1, $2, $3, $4) ON "
                                                     "CONFLICT (item_id, user_id) "
                                                     "DO UPDATE SET rating = EXCLUDED.rating, timestamp = EXCLUDED.timestamp")

    def get_records(self):
        for item_id, user_id, rating, timestamp in \
                self.connection.query("SELECT item_id, user_id, rating, timestamp FROM ratings"):
            yield ItemRating(user_id, item_id, rating, timestamp)

    def _ensure_items(self):
        if self._item_map is None:
            self._item_map = {}
            for item_id, name, genres in \
                    self.connection.query("SELECT item_id, name, genres FROM items"):
                self._item_map[item_id] = Item(item_id, name, genres.split(","))

    def get_item_description(self, item_id):
        self._ensure_items()

        if item_id in self._item_map:
            return str(self._item_map.get(item_id, "Unknown"))
        else:
            return "Unknown"

    def replace_items(self, items: Iterable[Item]):
        insert_items = self.connection.prepare("INSERT INTO items VALUES ($1, $2, $3) ON CONFLICT DO NOTHING")
        buffer = []
        for m in items:
            buffer.append((str(m.item_id), str(m.name)[:128], ",".join(m.genres)[:128]))
        insert_items.load_rows(buffer)

    def replace_ratings(self, ratings: Iterable[ItemRating], batch_size = 10000):
        buffer = []
        for r in ratings:
            buffer.append((r.item_id, r.user_id, r.rating, r.timestamp))
            if len(buffer) > batch_size:
                self._insert_items.load_rows(buffer)
                buffer = []
        if len(buffer) > 0:
            self._insert_items.load_rows(buffer)

    def get_items(self) -> Iterable[Tuple[str, Item]]:
        self._ensure_items()
        for k, i in self._item_map.items():
            yield k, i

    def put_record(self, item_id: str, user_id: str, rating: float, timestamp: int):
        print("Inserting the record")
        self._insert_items(item_id, user_id, rating, timestamp)

    def get_item_genres(self, item_id: str):
        return []

    def _init_db(self):
        self.connection.query("DROP TABLE IF EXISTS items")
        self.connection.query("DROP TABLE IF EXISTS ratings")
        self.connection.query("CREATE TABLE items(item_id varchar(16) PRIMARY KEY, name varchar(256), genres varchar(256))")
        self.connection.query("CREATE TABLE ratings(item_id varchar(16), user_id varchar(16), "
                              "rating float, timestamp int, PRIMARY KEY(item_id, user_id))")
