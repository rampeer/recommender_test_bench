import argparse
import logging

from flask import Flask, request, jsonify

from loader import MovieLensLoader
from recs import SVDBasedCF

app = Flask(__name__)


@app.route("/rest/<user_id>/recommend", methods=['GET', 'POST'])
def recommend_for_user(user_id):
    interests = rs.predict_interests(user_id)
    # Adding human-readable history and prediction interpretation for debug sake
    historical_records = [ldr.get_item_description(x.item_id) + " rated with " + str(x.rating)
                          for x in rs.user_histories.get(user_id, [])]
    predicted_interpretation = [ldr.get_item_description(x) for x in interests]
    response = {
        "recommendations": interests,
        "recommendation_names": predicted_interpretation,
        "history": historical_records
    }
    return jsonify(response)


@app.route("/rest/<user_id>/<item_id>/rate", methods=['GET', 'POST'])
def save_rate(user_id, item_id):
    try:
        rs.add_data(user_id, item_id, rating=float(request.args.get("rating", "5.0")))
        return jsonify({'ok': True})
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
