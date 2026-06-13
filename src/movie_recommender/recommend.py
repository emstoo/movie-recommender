"""Rating-weighted content-based recommendation.

We build a single "taste profile" vector by summing each rated title's feature
vector weighted by its mean-centered rating. Titles you rated above your
average pull the profile toward their content; titles below average push it
away. A candidate is then scored by cosine similarity to that profile.

This uses ``Your Rating`` — the original ``movie_distance.py`` ignored it
entirely and produced pure content distance with no notion of taste.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from .features import FeatureMatrix


def build_taste_profile(
    features: FeatureMatrix,
    ratings: pd.Series,
) -> np.ndarray:
    """Return an L2-normalized taste-profile vector (1 × n_features).

    ``ratings`` is a Const-indexed numeric series (``Your Rating``). Titles
    without a rating are ignored. Weights are mean-centered so disliked titles
    contribute negatively.
    """
    rated = ratings.dropna()
    common = features.index.intersection(rated.index)
    if len(common) == 0:
        raise ValueError("No rated titles overlap the feature matrix.")

    positions = [features.index.get_loc(c) for c in common]
    weights = rated.loc[common].to_numpy(dtype=np.float64)
    weights = weights - weights.mean()  # center around the user's average
    if np.allclose(weights, 0.0):
        # All ratings identical: fall back to an unweighted mean of content.
        weights = np.ones_like(weights)

    submatrix = features.matrix[positions]  # (k × n_features), sparse
    profile = submatrix.T @ weights  # (n_features,)
    profile = np.asarray(profile).ravel().reshape(1, -1)
    return normalize(profile, norm="l2", axis=1)


def recommend(
    features: FeatureMatrix,
    ratings: pd.Series,
    candidate_index: pd.Index | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """Rank candidates by similarity to the taste profile.

    ``candidate_index`` selects which rows of ``features`` to score. When None,
    every title that is *not* already rated is treated as a candidate (useful
    once TMDB-sourced unwatched titles are appended to the feature matrix).
    """
    profile = build_taste_profile(features, ratings)

    if candidate_index is None:
        rated_ids = ratings.dropna().index
        candidate_index = features.index.difference(rated_ids)

    positions = [features.index.get_loc(c) for c in candidate_index]
    if not positions:
        return pd.DataFrame(columns=["score"]).rename_axis("Const")

    candidate_matrix: sparse.csr_matrix = features.matrix[positions]
    scores = (candidate_matrix @ profile.T).ravel()

    result = pd.DataFrame({"score": scores}, index=pd.Index(candidate_index, name="Const"))
    return result.sort_values("score", ascending=False).head(top_n)
