"""Build an unwatched candidate pool from TMDB.

The IMDb export only contains titles you have already rated, so there is nothing
to recommend from it directly. Here we ask TMDB for titles it recommends
alongside your favourites, drop anything you have already watched, and fetch the
same genre+keyword features so candidates live in the same feature space as your
rated titles.

Candidates are keyed by their real IMDb id (``Const``) so they share the watched
titles' namespace; the TMDB id is kept alongside in the output.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from .tmdb import TmdbClient

logger = logging.getLogger(__name__)

# Fallback key for the rare candidate TMDB has no IMDb mapping for.
def _candidate_key(media_type: str, tmdb_id: int) -> str:
    return f"tmdb-{media_type}-{tmdb_id}"


def _select_seeds(enriched: pd.DataFrame, min_rating: float, max_seeds: int) -> pd.DataFrame:
    seeds = enriched[enriched["tmdb_resolved"] & enriched["tmdb_id"].notna()]
    if "Your Rating" in seeds.columns:
        seeds = seeds[seeds["Your Rating"] >= min_rating]
        seeds = seeds.sort_values("Your Rating", ascending=False)
    return seeds.head(max_seeds)


def build_candidate_pool(
    enriched: pd.DataFrame,
    client: TmdbClient,
    min_rating: float = 8.0,
    max_seeds: int = 20,
    max_candidates: int = 100,
    media_types: Optional[Set[str]] = None,
    languages: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Return a Const-indexed candidate frame with genres/keywords filled in.

    Columns mirror the enriched table (``Title``, ``genres``, ``tmdb_keywords``,
    ``tmdb_id``, ``media_type``) plus ``original_language`` and ``rec_count`` =
    how many of your seed titles recommended it.

    ``media_types`` restricts to {"movie", "tv"}; ``languages`` restricts to
    ISO-639-1 codes (e.g. {"en", "ja"}). ``None`` means no filter. Filtering
    happens before feature fetches, so excluded candidates cost no API calls.
    """
    watched_ids = set(enriched.loc[enriched["tmdb_id"].notna(), "tmdb_id"].astype(int))
    watched_imdb = set(enriched.index)
    seeds = _select_seeds(enriched, min_rating, max_seeds)
    logger.info("Gathering candidates from %d seed titles", len(seeds))

    # (media_type, tmdb_id) -> [title, rec_count, original_language]
    pool: Dict[Tuple[str, int], List] = {}
    for _, seed in seeds.iterrows():
        media_type = str(seed["media_type"])
        if media_types and media_type not in media_types:
            continue
        for rec in client.recommendations(int(seed["tmdb_id"]), media_type):
            rec_id = rec.get("id")
            if rec_id is None or rec_id in watched_ids:
                continue
            language = rec.get("original_language")
            if languages and language not in languages:
                continue
            key = (media_type, int(rec_id))
            title = rec.get("title") or rec.get("name") or ""
            if key in pool:
                pool[key][1] += 1
            else:
                pool[key] = [title, 1, language]

    # Most-recommended first, capped so we don't fire thousands of requests.
    ranked = sorted(pool.items(), key=lambda kv: kv[1][1], reverse=True)[:max_candidates]
    logger.info("Fetching features for %d candidates", len(ranked))

    rows = []
    for (media_type, tmdb_id), (title, rec_count, language) in ranked:
        genres, keywords, imdb_id = client.fetch_features(tmdb_id, media_type)
        # Prefer the real IMDb id as the key so candidates share the watched
        # titles' Const namespace; fall back to a synthetic key when TMDB has no
        # mapping. Skip anything that resolves to an already-watched title.
        if imdb_id and imdb_id in watched_imdb:
            continue
        const = imdb_id if imdb_id else _candidate_key(media_type, tmdb_id)
        rows.append(
            {
                "Const": const,
                "Title": title,
                "genres": genres,
                "tmdb_keywords": keywords,
                "tmdb_id": tmdb_id,
                "media_type": media_type,
                "imdb_id": imdb_id,
                "original_language": language,
                "rec_count": rec_count,
            }
        )

    columns = ["Title", "genres", "tmdb_keywords", "tmdb_id",
               "media_type", "imdb_id", "original_language", "rec_count"]
    if not rows:
        return pd.DataFrame(columns=columns).rename_axis("Const")
    return pd.DataFrame(rows).set_index("Const")
