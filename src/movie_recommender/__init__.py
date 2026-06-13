"""Content-based movie recommender built from an IMDb rating export.

Pipeline:
    1. enrich    – map each rated title to TMDB and fetch genres + keywords
    2. features  – turn genres/keywords into a sparse feature matrix
    3. similarity – cosine similarity between every pair of titles (Const-keyed)
    4. recommend – rank candidates by rating-weighted similarity to liked titles
"""

__all__ = ["__version__"]

__version__ = "0.2.0"
