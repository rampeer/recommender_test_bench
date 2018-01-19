from typing import List, Tuple, Dict, Iterable, Set
import numpy as np

from recs.recommender_engine import ItemRating
from .recommender_engine import RecommenderEngine
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Dot, Embedding, Flatten, Add, Concatenate
from keras.regularizers import l2

class ANNRecs(RecommenderEngine):
    """
    This recommender system is inspired by Neural CF
    He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). Neural Collaborative Filtering. https://doi.org/10.1145/3038912.3052569
    The only difference between model described in paper and the one implemented here is that ours solves
    regression task and minimizes MSE while the one in the paper is used as a classifier and minimizes logloss.
    This change is required because currently we assume that our goal is rating prediction.
    """
    def __init__(self, batch_size: int = 64,
                 user_embedding_size: int = 64, item_embedding_size: int = 64,
                 dense_sizes: List[int] = None,
                 epochs: int = 100, lr: float = .01, decay: float=1e-6,
                 validation_size: float = 0.1):
        if dense_sizes is None:
            dense_sizes = [32, 16, 8]
        self.batch_size = batch_size
        self.user_embedding_size = user_embedding_size
        self.item_embedding_size = item_embedding_size
        self.dense_sizes = dense_sizes
        self.epochs = epochs
        self.decay = decay
        self.lr = lr

        self.model = None  # type: Model
        self.user_indices = {}  # type: Dict[str, int]
        self.user_indices_reverse = {}  # type: Dict[int, str]
        self.item_indices = {}  # type: Dict[str, int]
        self.item_indices_reverse = {}  # type: Dict[str, int]
        self.user_train_items = {} # type: Dict[str, List[ItemRating]]
        self.user_validation_items = {} # type: Dict[str, List[ItemRating]]
        self.validation_size = validation_size
        # Users with validation interactions
        self.validation_users = set() # type: List[str]

        super().__init__()

    def _train_batch_generator(self) -> Iterable[Tuple[List, Iterable]]:
        items = np.zeros(self.batch_size)
        ratings = np.zeros(self.batch_size)
        while True:
            users = np.random.randint(0, len(self.user_indices), self.batch_size)
            for i, user_id_index in enumerate(users):
                history = self.user_train_items[self.user_indices_reverse[user_id_index]]
                random_entry = history[np.random.choice(len(history))]
                items[i] = self.item_indices[random_entry.item_id]
                ratings[i] = random_entry.rating
            yield [users, items], ratings

    def _validation_batch_generator(self) -> Iterable[Tuple[List, Iterable]]:
        items = np.zeros(self.batch_size)
        ratings = np.zeros(self.batch_size)
        while True:
            users = np.random.randint(0, len(self.validation_users), self.batch_size)
            for i, user_id in enumerate(users):
                history = self.user_validation_items[self.validation_users[user_id]]
                random_entry = history[np.random.choice(len(history))]
                items[i] = self.item_indices[random_entry.item_id]
                ratings[i] = random_entry.rating
            yield [users, items], ratings

    def _get_model(self) -> Model:
        input_user = Input(shape=(1,))
        input_item = Input(shape=(1,))

        user_emb = Embedding(len(self.user_histories), self.user_embedding_size, input_length=1, embeddings_regularizer=l2(1e-6))
        item_emb = Embedding(len(self.item_histories), self.item_embedding_size, input_length=1, embeddings_regularizer=l2(1e-6))

        current_layer = Concatenate()([
            Flatten()(user_emb(input_user)),
            Flatten()(item_emb(input_item))
        ])
        for i in self.dense_sizes:
            current_layer = Dense(i, activation="relu")(current_layer)
        output = Dense(1)(current_layer)
        model = Model(inputs=[input_user, input_item], outputs=output)

        return model

    def build(self):
        history_size = 0
        self.validation_users = []
        self.user_indices = {}
        for i, j in enumerate(self.user_histories):
            self.user_indices[j] = i
            self.user_indices_reverse[i] = j
            self.user_validation_items[j] = np.random.choice(
                    self.user_histories[j],
                    int(np.floor(len(self.user_histories[j]) * self.validation_size))
                )
            if len(self.user_validation_items[j]) > 0:
                self.validation_users.append(j)
            self.user_train_items[j] = list(set(self.user_histories[j]) - set(self.user_validation_items[j]))
            history_size += len(self.user_histories[j])
        self.item_indices = {}
        for i, j in enumerate(self.item_histories):
            self.item_indices[j] = i
            self.item_indices_reverse[i] = j
        self.model = self._get_model()
        self.model.compile(optimizer="adam", loss="mse")

        self.model.optimizer.lr = self.lr
        self.model.optimizer.decay = self.decay

        if self.validation_size is None or self.validation_size  == 0.0:
            self.model.fit_generator(self._train_batch_generator(),
                                     steps_per_epoch=history_size / self.batch_size,
                                     epochs=self.epochs)
        else:
            self.model.fit_generator(self._train_batch_generator(),
                                     steps_per_epoch=history_size / self.batch_size,
                                     epochs=self.epochs,
                                     validation_data=self._validation_batch_generator(),
                                     validation_steps=100)

    def predict_rating(self, user_id: str, item_id: str) -> float:
        if user_id not in self.user_indices:
            raise Exception("")
        if item_id not in self.item_indices:
            return float(np.mean([x.rating for x in self.user_histories[user_id]]))
        prediction = self.model.predict([
            np.array([self.user_indices[user_id]]),
            np.array([self.item_indices[item_id]])
        ])
        return prediction[0]
