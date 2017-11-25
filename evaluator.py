import numpy as np
import tqdm
from typing import List, Dict, Callable, Tuple
import operator

from loader import Loader
from recs.recommender_engine import RecommenderEngine, ItemRating


class Evaluator:
    # Utility functions. Exposing them, so caller can pass them to evaluate_* functions

    @staticmethod
    def ranking_metric_hit_rate(self, true_list: List[ItemRating], recommended_list: List[str]):
        if len(set(recommended_list) & set([x.item_id for x in true_list])) > 0:
            return 1.0
        else:
            return 0.0

    @staticmethod
    def ranking_metric_ndcg(self, true_list: List[ItemRating], recommended_list: List[str]):
        # We assume that relevance == rating. So, even if user rated movie 1.0, it is better than
        # recommending item user have not seen.
        if len(recommended_list) == 0:
            return 0.0
        relevance_map = {x.item_id: x.rating for x in true_list}

        recommendations_ratings = np.array([relevance_map.get(x, 0) for x in recommended_list])
        max_ratings = np.array([x.rating for x in true_list]).sort()[::-1][:len(recommended_list)]
        weights = 1.0 / np.log2(np.arange(len(recommended_list) + 1))

        return sum(recommendations_ratings * weights) / sum(max_ratings * weights)

    def _feed_partition(self, re: RecommenderEngine, partition: str):
        raise NotImplementedError

    def _evaluate_ranking_on_partition(self, re: RecommenderEngine, partition: str,
                                       scorers: List[Callable] = None) -> List[float]:
        raise NotImplementedError

    def _evaluate_scoring_on_partition(self, re: RecommenderEngine, partition: str,
                                       scorers: List[Callable] = None) -> List[float]:
        raise NotImplementedError

    def evaluate_ranking(self, re: RecommenderEngine, scorers: List[Callable] = None,
                         train_partitions: List[str] = ["train"], test_partitions: List[str] = ["test"]):
        for p in train_partitions:
            self._feed_partition(re, p)
        re.build()
        scoring = []
        for p in test_partitions:
            scoring.extend(self._evaluate_ranking_on_partition(re, p, scorers))
        return np.mean(np.array(scoring), axis=0)

    def evaluate_scoring(self, re: RecommenderEngine, scorers: List[Callable] = None,
                         train_partitions: List[str] = ["train"], test_partitions: List[str] = ["test"]):
        for p in train_partitions:
            self._feed_partition(re, p)
        re.build()
        scoring = []
        for p in test_partitions:
            scoring.extend(self._evaluate_scoring_on_partition(re, p, scorers))
        return np.mean(np.array(scoring), axis=0)


class TimeBasedEvaluator(Evaluator):
    """
    This evaluator emulates scenario where RE should provide recommendations to user with existing history.
    Events are fed to recommender in time-based fashion (for each user).
    This evaluator guarantees that each user in test set will have at least one record in train set.
    """

    def __init__(self, loader: Loader, partitions: List[Tuple[str, float]]=None):
        super().__init__()
        if partitions is None:
            partitions = [("train", 0.8), ("test", 0.2)]

        by_user = dict()  # type: Dict[str, List[ItemRating]]
        # We have to load records in memory to sort
        recs = list(loader.get_records())  # type: List[ItemRating]
        recs.sort(key=operator.attrgetter("timestamp"))

        for rec in recs:
            if rec.user_id not in by_user:
                by_user[rec.user_id] = []
            by_user[rec.user_id].append(rec)

        self.data_partitions = dict({p: [] for p, _ in partitions})  # type: Dict[str, List[ItemRating]]
        sliding_ratio = 0.0
        for p, ratio in partitions:
            for u, items in by_user.items():
                from_index = int(np.ceil(len(items) * sliding_ratio))
                to_index = int(np.floor(len(items) * (sliding_ratio+ratio)))
                self.data_partitions[p].extend(items[from_index:to_index])
            sliding_ratio += ratio

    def _feed_partition(self, re: RecommenderEngine, partition: str):
        for rec in self.data_partitions[partition]:
            re.add_data(rec.user_id, rec.item_id, rec.rating, rec.timestamp)

    def _evaluate_scoring_on_partition(self, re: RecommenderEngine, partition: str,
                                       scorers: List[Callable] = None):
        scores = []
        for rec in self.data_partitions[partition]:
            prediction = re.predict_rating(rec.user_id, rec.item_id)
            scores.append([scorer(rec.rating, prediction) for scorer in scorers])
            re.add_data(rec.user_id, rec.item_id, rec.rating, rec.timestamp)
        return scores
