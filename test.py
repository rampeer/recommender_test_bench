import random
from collections import Counter

import numpy as np

from evaluator import TimeBasedEvaluator
from loader import MovieLensLoader
from recs import SVDBasedCF, AverageRatingRecs, ANNRecs
from recs.user_cf import UserBasedNNCF
import argparse

random.seed(42)
ldr = MovieLensLoader("data/movielens/", 500000)
ev = TimeBasedEvaluator(ldr, [("train", 0.7), ("valid", 0.15), ("test", 0.15)])


def squared_error(a, b):
    return (a - b) * (a - b)


def absolute_error(a, b):
    return np.abs((a - b))

def cv_ann():
    print("ANN CV")
    cv_results_ann = Counter()
    ann_param_map = {}
    i = 0
    for last_layer_size in [8, 16, 32, 64]:
        for lr in [0.0001, 0.0005, 0.001, 0.005]:
            i += 1
            params = {"item_embedding_size": last_layer_size * 2,
                      "user_embedding_size": last_layer_size * 2,
                      "dense_sizes": [last_layer_size * 4, last_layer_size * 2, last_layer_size],
                      "lr": lr
                      }
            recs = ANNRecs(**params, epochs=50)
            score = ev.evaluate_scoring(recs,
                                        scorers=[squared_error, absolute_error], test_partitions=["valid"])
            print("ANN (%d %f) %f %f" % (last_layer_size, lr, score[0], score[1]))
            cv_results_ann[i] = -score[0]
            ann_param_map[i] = params
    return ann_param_map[cv_results_ann.most_common(1)[0][0]]

def cv_cf():
    print("Performing CV for UserBasedCF")
    cv_results_user_based = Counter()
    for n in [3, 5, 7, 10, 15, 20]:
        for correction in [UserBasedNNCF.CORRECTION_USER_MEAN, UserBasedNNCF.CORRECTION_NONE]:
            for prediction in [UserBasedNNCF.PREDICTION_AVERAGE, UserBasedNNCF.PREDICTION_UNBIASED_AVERAGE]:
                score = ev.evaluate_scoring(UserBasedNNCF(n, correction, prediction, n * 2),
                                            scorers=[squared_error, absolute_error], test_partitions=["valid"])
                print("UserCF(", n, correction, prediction, ")", score)
                cv_results_user_based[(n, correction, prediction)] = -score[0]
    return cv_results_user_based.most_common(1)[0][0]

def cv_svd():
    print("Performing CV for SVD")
    cv_results_svd = Counter()
    for i in range(10, 190, 15):
        score = ev.evaluate_scoring(SVDBasedCF(i), scorers=[squared_error, absolute_error], test_partitions=["valid"])
        print("SVD(", i, ")", score)
        cv_results_svd[i] = -score[0]  # Using MSE score to make final decision
    return cv_results_svd.most_common(1)[0][0]

def main(do_cf, do_svd, do_ann):
    final_eval_params = {
        "scorers": [squared_error, absolute_error],
        "train_partitions": ["train", "valid"],
        "test_partitions": ["test"]
    }
    print("=" * 10, "Final scores", "=" * 10)
    print("Average rating recs", ev.evaluate_scoring(AverageRatingRecs(), **final_eval_params))

    if do_svd:
        best_params = cv_svd()
        print("=" * 10, "Final scores", "=" * 10)
        print("SVD(", best_params, ")", ev.evaluate_scoring(SVDBasedCF(best_params), **final_eval_params))

    if do_cf:
        best_params = cv_cf()
        print("=" * 10, "Final scores", "=" * 10)
        print("UserBasedCF(", best_params, ")", ev.evaluate_scoring(UserBasedNNCF(*best_params), **final_eval_params))

    if do_ann:
        best_params = cv_ann()
        print("=" * 10, "Final scores", "=" * 10)
        print("ANN(", best_params, ")", ev.evaluate_scoring(ANNRecs(**dict(best_params)), **final_eval_params))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_ann", type=str2bool, nargs='?', const=True, default="no",
                        help="Do not evaluate Neural Collaborative Filtering")
    parser.add_argument("--skip_cf", type=str2bool, nargs='?', const=True, default="no",
                        help="Do not evaluate used-based collaborative filtering")
    parser.add_argument("--skip_svd", type=str2bool, nargs='?', const=True, default="no",
                        help="Do not evaluate SVD-based recommender")
    (args) = parser.parse_args()

    main(not args.skip_cf, not args.skip_svd, not args.skip_ann)
