import random
from collections import Counter

import numpy as np

from evaluator import TimeBasedEvaluator
from loader import MovieLensLoader
from recs import SVDBasedCF, AverageRatingRecs

random.seed(42)
ldr = MovieLensLoader("data/movielens/", 10000000)
ev = TimeBasedEvaluator(ldr, [("train", 0.7), ("valid", 0.15), ("test", 0.15)])


def squared_error(a, b):
    return (a - b) * (a - b)


def absolute_error(a, b):
    return np.abs((a - b))

print("Performing CV for SVD")

cv_results = Counter()
for i in range(10, 190, 15):
    score = ev.evaluate_scoring(SVDBasedCF(i), scorers=[squared_error, absolute_error], test_partitions=["valid"])
    print("SVD(", i, ")", score)
    cv_results[i] = -score[0]  # Using MSE score to make final decision

final_eval_params = {
    "scorers": [squared_error, absolute_error],
    "train_partitions": ["train", "valid"],
    "test_partitions": ["test"]
}
print("=" * 10, "Final scores", "=" * 10)

print("Average rating recs", ev.evaluate_scoring(AverageRatingRecs(), **final_eval_params))
best_n = cv_results.most_common(1)[0][0]
print("SVD(", best_n, ")", ev.evaluate_scoring(SVDBasedCF(best_n), **final_eval_params))
