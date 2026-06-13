"""Load and normalize the IMDb rating export.

The IMDb "Your Ratings" CSV export has a stable column layout; we keep the
columns the recommender actually needs and normalize their dtypes so the rest
of the pipeline never has to second-guess the input.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

# Columns we rely on downstream. "Const" (the tt-id) is the stable join key.
REQUIRED_COLUMNS = ["Const", "Title", "Year", "Genres"]
OPTIONAL_COLUMNS = ["Your Rating", "Title Type", "Original Title"]


class RatingsError(RuntimeError):
    """Raised when the ratings export is missing or malformed."""


def split_genres(raw_value: object) -> List[str]:
    """Split IMDb's comma-separated ``Genres`` cell into a clean token list."""
    if pd.isna(raw_value):
        return []
    parts = [part.strip() for part in str(raw_value).split(",")]
    return [p for p in parts if p]


def load_ratings(csv_path: Path) -> pd.DataFrame:
    """Read the IMDb export and return a normalized frame indexed by Const.

    Guarantees on the returned frame:
        - index is ``Const`` (unique tt-id), as a string
        - ``Genres`` is a list[str] column named ``genre_list``
        - ``Your Rating`` is numeric (NaN when absent)
    """
    if not csv_path.exists():
        raise RatingsError(f"Ratings export not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"Const": "string"})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RatingsError("Missing required column(s): " + ", ".join(missing))

    df = df.dropna(subset=["Const"]).copy()
    df["Const"] = df["Const"].str.strip()
    if df["Const"].duplicated().any():
        dupes = df.loc[df["Const"].duplicated(), "Const"].tolist()
        raise RatingsError(f"Duplicate Const values in export: {dupes}")

    df["genre_list"] = df["Genres"].apply(split_genres)
    if "Your Rating" in df.columns:
        df["Your Rating"] = pd.to_numeric(df["Your Rating"], errors="coerce")

    return df.set_index("Const")
