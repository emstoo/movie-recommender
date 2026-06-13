"""Pairwise cosine similarity between titles.

Replaces ``movie_distance.py``. Key changes from the original:
    - Keyed by ``Const`` (tt-id), not by a sanitized title+year filename.
    - Outputs a single square similarity frame, not 99 per-movie CSVs.
    - No per-row min-max normalization. That step made scores incomparable
      across titles (every row was force-stretched to span [0, 1]); raw cosine
      similarity in [0, 1] is already meaningful and comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FeatureMatrix


def cosine_similarity_frame(features: FeatureMatrix) -> pd.DataFrame:
    """Return an N×N cosine-similarity DataFrame indexed and columned by Const.

    Rows of ``features.matrix`` are already L2-normalized, so the Gram matrix
    (X · Xᵀ) is exactly the cosine similarity. Values lie in [0, 1] because all
    features are non-negative.
    """
    gram = (features.matrix @ features.matrix.T).toarray()
    np.clip(gram, 0.0, 1.0, out=gram)
    # Numerical guarantee: the diagonal (self-similarity) is exactly 1 for any
    # title that has at least one feature, 0 for a featureless one.
    return pd.DataFrame(gram, index=features.index, columns=features.index)


def nearest_neighbors(
    similarity: pd.DataFrame,
    const: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """Most similar titles to ``const``, excluding the title itself."""
    if const not in similarity.index:
        raise KeyError(f"Unknown Const: {const}")
    scores = similarity.loc[const].drop(index=const)
    ranked = scores.sort_values(ascending=False).head(top_n)
    return ranked.rename("similarity").to_frame()
