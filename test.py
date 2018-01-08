import random
from collections import Counter

import numpy as np

from evaluator import TimeBasedEvaluator
from loader import MovieLensLoader
from recs import SVDBasedCF, AverageRatingRecs
from recs.user_cf import UserBasedNNCF

random.seed(42)
ldr = MovieLensLoader("data/movielens/", 500000)
ev = TimeBasedEvaluator(ldr, [("train", 0.7), ("valid", 0.15), ("test", 0.15)])


def squared_error(a, b):
    return (a - b) * (a - b)


def absolute_error(a, b):
    return np.abs((a - b))


print("Performing CV for UserBasedCF")
cv_results_user_based = Counter()
for n in [3, 5, 7, 10, 15, 20]:
    for correction in [UserBasedNNCF.CORRECTION_USER_MEAN, UserBasedNNCF.CORRECTION_NONE]:
        for prediction in [UserBasedNNCF.PREDICTION_AVERAGE, UserBasedNNCF.PREDICTION_UNBIASED_AVERAGE]:
            score = ev.evaluate_scoring(UserBasedNNCF(n, correction, prediction, n * 2),
                                        scorers=[squared_error, absolute_error], test_partitions=["valid"])
            print("UserCF(", n, correction, prediction, ")", score)
            cv_results_user_based[(n, correction, prediction)] = -score[0]


print("Performing CV for SVD")
cv_results_svd = Counter()
for i in range(10, 190, 15):
    score = ev.evaluate_scoring(SVDBasedCF(i), scorers=[squared_error, absolute_error], test_partitions=["valid"])
    print("SVD(", i, ")", score)
    cv_results_svd[i] = -score[0]  # Using MSE score to make final decision


final_eval_params = {
    "scorers": [squared_error, absolute_error],
    "train_partitions": ["train", "valid"],
    "test_partitions": ["test"]
}
print("=" * 10, "Final scores", "=" * 10)

print("Average rating recs", ev.evaluate_scoring(AverageRatingRecs(), **final_eval_params))
best_params = cv_results_svd.most_common(1)[0][0]
print("SVD(", best_params, ")", ev.evaluate_scoring(SVDBasedCF(best_params), **final_eval_params))
best_params = cv_results_user_based.most_common(1)[0][0]
print("UserBasedCF(", best_params, ")", ev.evaluate_scoring(UserBasedNNCF(*best_params), **final_eval_params))
