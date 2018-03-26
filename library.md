### BPR: Bayesian Personalized Ranking from Implicit Feedback

Most the real-world feedback is implicit; implicit data is abundant and rich. Authors introduce optimization method (BPR-Opt) that fits model to maximize likelihood of good product to be ranked better than bad one. This method is closely related to AUC optimization. Training set is triplets of "good" and "bad" products and users. Good products are the ones that user has interacted with; bad products are all other products. Due to catalog size, stochastic gradient is used.

LearnBPR is applied to Matrix Factorization problem. SVD is guaranteed to optimize square error, but is prone to overfitting. Other methods such as non-negative factorization, maximum margin factorization are usually used to tackle this problem.

Adaptive k-Nearest-Neighbours are also used with conjunction with BPR-Opt. The optimization criterion is used to estimate similarity measure between items (instead of heuristic Pearson correlation).

Weighted Regularized Matrix Factorization and Maximum Margin Matrix Factorization are used as baselines. AUC is used to estimate recommendations quality; for test sets a single interaction was hidden for each user.

It is shown that models with BPR-Opt outperform baselines and theoretical maximal non-personalized recommender performance.




### Bayesian Personalized Ranking with Multi-Channel User Feedback

https://www.youtube.com/watch?v=aKHLf4P3N08

MF-BPR is extension of Bayesian Personalized Ranking with multi (channel) feedback. User feedback is drawn from different channels (event types), unlike the conventional BPR. BPR is a pairwise L2Rank models that uses positive and negative sampling; this sampling procedure greatly affects performance.

Standard BPR sampling: getting triple (u, i, j) using observed positive feedback (u, i) and negative item j. Model should rank items like that: (observed > unobserved)
MF-BPR sampling: (purchase > a2cart > click > unobserved >  negative). Positive samples are sampled non-uniformly; negative samples can be sampled uni or non-uni.

Authors compared: BPR, BPR-Dynamic, MF-BPR-UNI (uniform sampler), MF-BPR (non-uniform sampler). MRR is used to evaluate recommendations quality. Interesting result is that sampling heavily influences performance, and non-uniform sampler with MF-BPR performs the best.



### GroupLens: An Open Architecture for Collaborative Filtering of Netnews

The paper is concerned about a system (servers) called Better Bit Bureaus that collect news rating data and provide users with news recommendations.

The rating storage is distributed. User ratings are packed into special "news article" that is posted onto local Usenet server; that article afterwards is propagated to others servers. That creates several implications such as data duplication and storage overhead, but that propagation scheme is easy to implement the prototype with tech that was available those days.

BBB differs from existing recommender systems of that time (which were content-based). The prediction algorithm relies on ratings. Authors note that they have tried different techniques such as reinforcement learning (!), multivariate regression, and pairwise correlation coefficients; but only pairwise prediction approach is explained in details. In order to make prediction for user U on item I:

1) User is compared with others using Pearson correlation on co-rated items.
2) For each user that has rated item I, subtract average rating for that user from rating of I given by that user, and take weighted average using correlations from step 1 as weight.
3) Add user's U average rating to that weighted average.

Authors have not ran or designed any test to confirm that their system works better than existing one (sorting news by recency). Instead, they implemented and ran the system to find any scalability issues. Also, authors note that such system would have social impact: need for moderated newsgroup would decline; large newsgroups would not be required to split as their traffic increases; and cumbersome filtering systems would not be required no more. Also, the paper foresees problem of filtering bubble.


### Reranking Strategies Based on Fine-Grained Business User Events Benchmarked on a Large E-commerce Data Set

Authors compare different non-personalized reranking techniques that would improve text search.

Real industrial data provided by Cdiscount is used. Each click, add to cart and purchase is assigned with last query typed by user. Top 1000 queries are considered (it is not clear whether filtering was done before or after assignation).

Following models are considered:
1. Random ranking
2. BM 25 (20 year old state of the art)
3. Reranking by click count (last month)
4. Reranking by projected revenue

In order to project revenue, authors use auto regressive model that takes clicks, a2c and purchases for the last P days, and tries to fit linear regularized mixture to approximate number of purchases for current day. To convert number of purchases into revenue, author multiply the number by item price. Authors fitted autoregression using all available time series and from the first date (I.e. Maximum P?)

Quality metric is percent of revenue that falls in top K items (percent of total revenue).

It's unclear how candidates were chosen; however, paper hints that a basic keyword search was applied for that procedure. Also, normalization procedure of the quality metric is not well-descripted.

The result is that projected revenue works best; reranking by click count works slightly worse; and BM model is marginally better than random shuffle.
