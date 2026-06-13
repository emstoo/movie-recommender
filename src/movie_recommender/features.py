"""Turn genres + keywords into a per-title feature matrix.

We build one binary token per genre and per keyword, namespaced so a genre and
a keyword that share a word never collide (``g:Drama`` vs ``k:drama``). Genres
are coarse (~20 of them) while keywords are specific and numerous, so genres get
a configurable weight to keep them from being drowned out.

Rows are L2-normalized, which means a plain dot product equals cosine
similarity — handy for the similarity and recommendation steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer, normalize


@dataclass
class FeatureMatrix:
    """L2-normalized sparse features keyed by Const index."""

    index: pd.Index  # Const values, row-aligned with ``matrix``
    matrix: sparse.csr_matrix
    vocabulary: List[str]


def _tokens(genres: Sequence[str], keywords: Sequence[str]) -> List[str]:
    out = [f"g:{g}" for g in genres if g]
    out += [f"k:{k}" for k in keywords if k]
    return out


def build_features(
    df: pd.DataFrame,
    genre_weight: float = 3.0,
) -> FeatureMatrix:
    """Build an L2-normalized feature matrix from ``genres``/``tmdb_keywords``.

    ``df`` must be Const-indexed with list columns ``genres`` and
    ``tmdb_keywords`` (see :mod:`movie_recommender.enrich`).
    """
    token_docs = [
        _tokens(row.get("genres") or [], row.get("tmdb_keywords") or [])
        for _, row in df.iterrows()
    ]

    mlb = MultiLabelBinarizer(sparse_output=True)
    binary = mlb.fit_transform(token_docs).astype(np.float64).tocsr()

    if binary.shape[1] == 0:
        # No genres and no keywords anywhere: a single zero column keeps shapes
        # valid; every similarity will be 0, which is the honest answer.
        binary = sparse.csr_matrix((binary.shape[0], 1), dtype=np.float64)
        vocabulary: List[str] = ["__empty__"]
    else:
        vocabulary = list(mlb.classes_)
        # Up-weight genre columns before normalization.
        if genre_weight != 1.0:
            weights = np.array(
                [genre_weight if v.startswith("g:") else 1.0 for v in vocabulary]
            )
            binary = binary.multiply(weights).tocsr()

    normalized = normalize(binary, norm="l2", axis=1)
    return FeatureMatrix(index=df.index, matrix=normalized, vocabulary=vocabulary)
