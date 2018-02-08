all: build
	docker-compose up

fill-db: build
	docker-compose run application python3 init_db.py

test: build
	docker-compose run application python3 test.py

build:
	docker-compose build
