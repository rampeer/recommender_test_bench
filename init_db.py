#insert_person = db.prepare("INSERT INTO person VALUES ($1, $2, $3, $4, $5, $6, $7, $8) "
#                           "ON CONFLICT DO NOTHING")

from loader import PostgresLoader
from loader import MovieLensLoader

source = MovieLensLoader("data/movielens/", 2000000)
destination = PostgresLoader("postgres", "postgres", "rs_pg", 5432, "mydb", True)

destination.replace_items(source.movies.values())

destination.replace_ratings(source.get_records())
