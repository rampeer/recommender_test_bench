from collections import Counter
from recs import RecommenderEngine


class HeuristicRE(RecommenderEngine):
    def __init__(self):
        super().__init__()
        self.item_item_influence = None

    def build(self):
        self.item_item_influence = dict()
        item_interactions = Counter()
        for _, ratings in self.user_histories.items():
            if len(ratings) > 50:
                continue
            if len(ratings) == 0:
                continue
            user_avg_rating = sum([r.rating for r in ratings]) / len(ratings)
            for r in ratings:
                for r2 in ratings:
                    if r != r2:
                        if r.item_id not in self.item_item_influence:
                            self.item_item_influence[r.item_id] = dict()
                        if r2.item_id not in self.item_item_influence[r.item_id]:
                            self.item_item_influence[r.item_id][r2.item_id] = 0
                        self.item_item_influence[r.item_id][r2.item_id] += r2.rating - user_avg_rating
                        item_interactions[r.item_id] += 1

        for key, recs in self.item_item_influence.items():
            for k in recs.keys():
                recs[k] = recs[k] / item_interactions[key]

    def predict_interests(self, user_id: str):
        already_seen = set()
        recs = Counter()
        for r in self.user_histories.get(user_id, []):
            already_seen.add(r.item_id)
            for rec, value in self.item_item_influence.get(r.item_id, {}).items():
                recs[rec] += value

        for i in already_seen:
            recs.pop(i, None)

        return [x for _, x in recs.most_common(5)]
