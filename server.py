import argparse
import logging

from flask import Flask, request, jsonify

from loader import MovieLensLoader
from recs import SVDBasedCF

app = Flask(__name__)


@app.route("/rest/<user_id>/recommend", methods=['GET', 'POST'])
def recommend(user_id):
    interests = rs.predict_interests(user_id)
    predicted_interpretation = [(x, ldr.get_item_description(x)) for x in interests]
    response = {
        "recommendations": predicted_interpretation,
    }
    return jsonify(response)


@app.route("/rest/<user_id>/history", methods=['GET', 'POST'])
def show_history(user_id):
    historical_records = [ldr.get_item_description(x.item_id) + " rated with " + str(x.rating)
                          for x in rs.user_histories.get(user_id, [])]
    response = {
        "history": historical_records,
    }
    return jsonify(response)


@app.route("/rest/<user_id>/<item_id>/rate", methods=['POST'])
def rate(user_id, item_id):
    try:
        rs.add_data(user_id, item_id, rating=float(request.args.get("rating", "5.0")))
        return jsonify({'ok': True})
    except Exception as e:
        logging.exception("Exception during subscriber optimization")
        return jsonify({'ok': False, 'error': str(e)}), 400


@app.route("/rest/find_item/<query>/", methods=['POST', 'GET'])
def find_item(query: str):
    try:
        tokens = set(query.split())
        found = {}
        for item_id, item in ldr.movies.items():
            if len(set(item.name.split()) & tokens) == len(tokens):
                found[item_id] = str(item)
                if len(found) > 30:
                    break
        return jsonify(found)
    except Exception as e:
        logging.exception("Exception during subscriber optimization")
        return jsonify({'ok': False, 'error': str(e)}), 400


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Movie recommender engine')
    parser.add_argument("--port", default=80, help="Port for server")
    (args) = parser.parse_args()
    # Populating recommender system with data
    ldr = MovieLensLoader("data/movielens/", 10000000)
    rs = SVDBasedCF(70, incremental_update=True)
    for r in ldr.get_records():
        rs.add_data(r.user_id, r.item_id, r.rating, r.timestamp)
    rs.build()

    app.run("0.0.0.0", port=args.port)
