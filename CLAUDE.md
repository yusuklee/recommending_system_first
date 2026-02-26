# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A recommendation system learning project (추천 시스템 입문) using the MovieLens 10M dataset. Implements and evaluates different recommendation algorithms in Python. Comments and variable names mix Korean and English.

## Running Recommenders

Each recommender is implemented as a subclass of `BaseRecommender` with a `recommend()` method, and can be evaluated by calling `.eval()`:

```bash
python -c "from Random import RandomRecommender; RandomRecommender().eval()"
```

Jupyter notebooks serve as both experimentation scratchpads and recommender implementations.

## Implemented Recommenders

| Notebook | Class | Algorithm |
|----------|-------|-----------|
| `Random.ipynb` | `RandomRecommender` | Random baseline |
| `stats.ipynb` | `PopularityRecommender` | Popularity-based (average ratings) |
| `apriori.ipynb` | `AssociationRecommender` | Market basket analysis (mlxtend apriori) |
| `collaborative_filter.ipynb` | *(standalone, no BaseRecommender)* | User-based CF with Pearson correlation |
| `rfr_recommender.ipynb` | `RFRecommender` | RandomForestRegressor with user/movie stats + genre features |
| `randomForest.ipynb` | *(exploration notebook)* | RF feature engineering scratchpad |
| `sub.ipynb` | *(exploration notebook)* | Data loading and EDA |

## Architecture

**Recommender pattern:** All recommenders extend `BaseRecommender` (ABC) which provides an `eval()` method. Subclasses implement `recommend(dataset, **kwargs)` and return a `RecommendResult`.

**Data flow:**
1. `DataLoader` loads MovieLens `.dat` files (`::`-separated) from `ml-10M100K/`, merges ratings with movie metadata and tags
2. Data is split by timestamp: each user's most recent `test_size` ratings become test, the rest become train
3. `recommend()` receives a `Dataset` and returns `RecommendResult` with predicted ratings and top-k item lists per user
4. `MetricCalculator` evaluates predictions using RMSE, Recall@K, and Precision@K

**Key data models** (`util/models.py`, all frozen dataclasses):
- `Dataset`: train DataFrame, test DataFrame, `user_love_items` (dict of user_id -> movie_ids rated >= 4 in test), `item_content` (movies DataFrame with genre list + tags)
- `RecommendResult`: predicted rating Series/DataFrame, `user_love_items` (predicted top items per user)
- `Metrics`: rmse, recall, precision (with formatted `__repr__`)

**DataLoader** (`util/data_loader.py`): `DataLoader(user_size=1000, test_size=5, data_path="ml-10M100K")` — `user_size` limits number of users loaded, `test_size` sets how many recent ratings per user go to test set.

## Dependencies

- pandas, numpy, scikit-learn (for `mean_squared_error`)
- mlxtend (for `apriori.ipynb` only — association rules)

## Data

The `ml-10M100K/` directory is gitignored. It must contain `movies.dat`, `ratings.dat`, and `tags.dat` from the [MovieLens 10M dataset](https://grouplens.org/datasets/movielens/10m/). Files use `::` as separator and `latin-1` encoding.


