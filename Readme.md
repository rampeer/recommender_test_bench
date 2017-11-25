## Description

This repo contains recommender system demo.

You can launch it by running server.py:

```
python3.6 server.py
```

By default it launches on 80 port, but you may specify it:

```
python3.6 server.py --port 5757
```

## API

This service have two endpoints:

```
/rest/<user_id>/recommend
```

returns JSON with "recommendations" field which holds list of identifiers of recommended movies. 
It also contains "history" debug field with list of users ratings (may contain duplicates!) and "recommendation_names" with human-readable recommendation list. 

Second endpoint,

```
/rest/<user_id>/<movie_id>/rate?rating=<rating>
```

assigns a rating to the user. It may also triggers recommendations update.

## Recommender engine

[MovieLens dataset](http://files.grouplens.org/datasets/movielens/ml-latest.zip) is used. It is expected to be unzipped in data/movielens.

Baseline (average rating prediction) and relation between item rating prediction and Top-N ranking are described in [here](http://ieeexplore.ieee.org/document/5680904). 

Extensive description of SVD-based recommender system is available [here](http://www.dtic.mil/get-tr-doc/pdf?AD=ADA439541). In this paper average user rating is subtracted from each rating. However, [some authors](https://dl.acm.org/citation.cfm?id=1010618) show that we can subtract average item rating. We can even fit simple linear regression (i.e. subtract both users', items' and global affinities) and [use SVD on residuals](https://www.mimuw.edu.pl/~paterek/ap_kdd.pdf). I wonder whether anyone will read this carefully enough to notice this sentence. I've used item average subtraction because it worked well in my experience and provides more diverse recommendations than user average rating subtraction.

SVD update step I've inferred myself; a more general, industrial-level approach is described in  [this paper](https://pdfs.semanticscholar.org/02ff/37cd0059cf1af1ecfa62c32304c05ab3bf96.pdf). 

Also, I wish I had time to implement and use "golden standard", widely known [User-based and Item-based CF](https://cseweb.ucsd.edu/~jmcauley/cse255/reports/wi15/Guanwen%20Yao_Lifeng_Cai.pdf) and usem as baselines.
  
## Evaluation

Cross-validation is done by running

```
python3.6 test.py
```

Dataset is divided into train, validation and test sets (by time in order to more closely emulate real) in 70% / 15% / 15% proporions.

RMSE and MAE are used to evaluate recommender engine. If we would accurately predict user rating, then will accurately make Top-N recommendations. There is [controversy regarding MSE and MAE](https://medium.com/netflix-techblog/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429) for recommender system evaluation, but it's still widely used. I've decided to use it because rating prediction - based metrics are more "fine-grained" (i.e. small improvement in recommender system may not impact NDCG/AUC/Precision/Recall, but will be visible on MSE/MAE, which is especially good for cross-validation).

First, using train and validation sets I determine best number of component for SVD. Best number of components is 70.

Then, using (train+validation) and test sets I calculate final scoring:

|                     | MSE       | MAE        |
|---------------------|------------|------------|
| Average rating recs | 0.92242227 | 0.74030889 |
| SVD                 | **0.85277154** | **0.70372001** |

We can see that SVD performs better than baseline by 8% (MSE)