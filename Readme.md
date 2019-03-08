This repository contains implementation of different recommender algorithms, along with code to evaluate them against
MovieLens dataset. Currently implemented algorithms:

- User-Based Collaborating Filtering
- Neural Collaborative Filtering
- Matrix Factorization (SVD) based recommender


## Recommender algorithms

Currently, I consider only rating prediction-based recommenders.

There are other tasks such as making predictions for implicit actions (i.e. without direct ratings from users) and
top-N recommendations where we have to provide the list of recommendations. It is very wide and interesting area of research,
but currently, I'm just too fed up with them on my main job. Someday I will put them here, I promise.

### User-based collaborative filtering

[Source](recs/user_cf.py)

The holy staple of recommender systems, almost synonymous with them. For each user, this system finds N similar users
(using some similarity measure), and uses actions of this neighborhood to make predictions for the user.
The algorithm is described in details [here](https://doi.org/10.1145/192844.192905) and
[here](https://doi.org/10.1145/371920.372071).

Pearson correlation is used to measure user similarity, as it is commonly employed in the literature.
Rating average and weighted rating average may be used for prediction.
Several engineering techniques such as lookups and sampling were used to speed up model computation.
I haven't found any proper implementations of User-based CF, as authors tend to miss the fact that we
should take into account only co-rated items.

### SVD-based recommender system

[Source](recs/svd_based_recs.py)

Recommender system based on matrix factorization of the user-rating matrix. It can be viewed as assigning vectors to each
item and the user that way that dot product of these vectors would predict the rating (or deviance of the rating from
average, see details below). This recommender is somewhat related to collaborative filtering, as it also
"clusters" users together in some latent space, but it does that softly by projecting ratings.

Extensive description of SVD-based recommender system is available [here](http://www.dtic.mil/get-tr-doc/pdf?AD=ADA439541).
In this paper average user rating is subtracted from each rating. However,
[some authors](https://dl.acm.org/citation.cfm?id=1010618) show
that we can subtract average item rating. We can even fit simple linear regression (i.e. subtract both users', items'
and global affinities) and [use SVD on residuals](https://www.mimuw.edu.pl/~paterek/ap_kdd.pdf).
I've used item average subtraction because it worked well in my experience and provides more diverse recommendations than
user average rating subtraction.

SVD update step I've inferred myself; a more general, industrial-level approach is described in
[this paper](https://pdfs.semanticscholar.org/02ff/37cd0059cf1af1ecfa62c32304c05ab3bf96.pdf).

### Neural Collaborative Filtering

[Source](recs/ann_recs.py)

This model is somewhat similar to the SVD-based recommender systems. It also assigns a vector to each item and user, but
then it uses several stacked of fully-connected layers to make the prediction.

This code is straightforward implementation of the [paper](https://doi.org/10.1145/3038912.3052569). The only thing
I had to make is change task type from classification with logloss to regression with MSE.

### Product-based Neural Network

This model is described in this [paper](https://arxiv.org/pdf/1611.00144.pdf). 
The "product" means "inner/outer product" (not "shop item"). 

The core idea of this model is to map each field (feature) into an embedding, and calculate pairwise dot products of all
embeddings (with extra product with "1" vector for capturing of linear, non-quadratic effects). After that we pass 
the results through several dense layers. The output of the this model is a single value, probability.

Authors propose two flavours of this architecture: with inner (dot) product and with outer product.

Model can be applied to the recommendation task; however, as one may notice, without any extra user or item feature
the model becomes similar to ALS/SVD recommender system (there are two "features": used ID and item ID; and single item
interaction).

### Convolutional Click Prediction Model

[Paper](http://www.escience.cn/system/download/73676)

### Average rating

[Source](recs/average_rating_recs.py)

Predicts ratings using item average ratings. Commonly used as a baseline. It is described
[here](http://ieeexplore.ieee.org/document/5680904).

## Evaluation

[MovieLens dataset](http://files.grouplens.org/datasets/movielens/ml-latest.zip) is used. It is expected to be unzipped in data/movielens.

Cross-validation and evaluation is done by running.

```
python3.6 test.py
```

or (using Docker)

```
make fill-db
```

Dataset is divided into train, validation and test sets (by time in order to more closely emulate real) in 70% / 15% / 15% proportions.

RMSE and MAE are used to evaluate recommender engine. If we would accurately predict user rating, then will accurately make Top-N recommendations. There is [controversy regarding MSE and MAE](https://medium.com/netflix-techblog/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429) for recommender system evaluation, but it's still widely used. I've decided to use it because rating prediction - based metrics are more "fine-grained" (i.e. small improvement in recommender system may not impact NDCG/AUC/Precision/Recall, but will be visible on MSE/MAE, which is especially good for cross-validation).

First, using train and validation sets I determine best hyperparameter settings

Then, using (train+validation) and test sets I calculate final scoring:

|                     | MSE       | MAE        |
|---------------------|------------|------------|
| Average rating recs | 0.90255803 | 0.7326334 |
| SVD                 | 0.850431384* | 0.70807589 |
| CF                 | **0.80323668** | **0.673640151** |
| ANN CF                 | 0.957307 | 0.755213 |

We can see that CF with Pearson correlation performs best. It is quite strange because it is commonly used baseline and
every paper claims to beat it. Currently I have no idea where I have failed.

Please note that despite my CF implementation uses heuristic sampling, search space is very large, and this model is
much slower than competitors. Also, it does not support online learning (like SVD), and model has to be rebuilt from
scratch to use new interactions and be able to make predictions for new users. These effects will be captured with
other testing scenarios.

## Running the service

### Launching the service

You can launch whole service by running single

```
make
```

This will launch the whole service. Before that, you can populate the database with MovieLens dataset using

```
make fill-db
``` 

You also might want to edit docker-compose.yaml and Makefile to specify recommender engine or ports.

Alternatively, you may launch recommender system without Docker
by running server.py:

```
python3.6 server.py
```

By default it launches on 80 port, but you may specify it:

```
python3.6 server.py --port 5757
```

### API

This service has two primary endpoints:

```
/rest/<user_id>/recommend
```

returns JSON with recommendations for that particular user and

```
/rest/<user_id>/<movie_id>/rate?rating=<rating>
```

assigns a rating to the user. It may also triggers recommendations update.

Also, service has two debug endpoints:

```
/rest/<user_id>/history
```

returns user rating history.

```
/rest/search?query=<name>
```

Finds and returns item IDS by item names.
