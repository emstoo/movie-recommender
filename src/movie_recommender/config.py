"""Centralized paths and runtime configuration.

TMDB credentials are read from the environment so they never live in the repo.
Two auth styles are supported (get either at
https://www.themoviedb.org/settings/api):

    TMDB_API_TOKEN  – v4 "API Read Access Token" (long JWT, ``eyJ...``),
                      sent as ``Authorization: Bearer``. Preferred.
    TMDB_API_KEY    – v3 API key (short string), sent as an ``api_key`` query
                      param. Used when no token is set.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Project root = two levels up from this file (src/movie_recommender/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"

# Inputs / outputs.
RATINGS_CSV = RAW_DIR / "movies.csv"
ENRICHED_PARQUET = INTERIM_DIR / "enriched.parquet"
SIMILARITY_PARQUET = INTERIM_DIR / "similarity.parquet"
CANDIDATES_PARQUET = INTERIM_DIR / "candidates.parquet"

TMDB_API_KEY_ENV = "TMDB_API_KEY"
TMDB_API_TOKEN_ENV = "TMDB_API_TOKEN"


@dataclass(frozen=True)
class TmdbConfig:
    # Exactly one of these is set; the client picks the matching auth style.
    api_key: Optional[str] = None
    bearer_token: Optional[str] = None
    base_url: str = "https://api.themoviedb.org/3"
    # Be a polite API citizen; TMDB allows ~50 req/s but we stay well under.
    request_timeout: float = 15.0
    max_retries: int = 3
    backoff_seconds: float = 1.5


def load_tmdb_config() -> TmdbConfig:
    """Build TMDB config or raise loudly when no credential is set.

    Prefers the v4 bearer token. Failing here (rather than silently returning
    empty data) is deliberate: the previous cinemagoer-based pipeline swallowed
    every error and wrote empty keywords without anyone noticing.
    """
    token = os.environ.get(TMDB_API_TOKEN_ENV, "").strip()
    if token:
        return TmdbConfig(bearer_token=token)

    api_key = os.environ.get(TMDB_API_KEY_ENV, "").strip()
    if api_key:
        return TmdbConfig(api_key=api_key)

    raise RuntimeError(
        f"No TMDB credential found. Set either {TMDB_API_TOKEN_ENV} (v4 read "
        f"access token, preferred) or {TMDB_API_KEY_ENV} (v3 key). Get them at "
        "https://www.themoviedb.org/settings/api"
    )
