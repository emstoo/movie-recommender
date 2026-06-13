"""Enrich rated titles with TMDB genres + keywords.

Replaces the old ``add_keywords.py`` (cinemagoer), which silently produced
empty keywords for every title because IMDb scraping is broken.

Key difference: failures are counted and reported. If TMDB resolves zero
titles, the run exits non-zero instead of writing a uselessly empty table.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import pandas as pd

from .ratings import split_genres
from .tmdb import TmdbClient, TmdbError

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentReport:
    total: int = 0
    resolved: int = 0
    not_found: int = 0
    errored: int = 0

    @property
    def ok(self) -> bool:
        return self.resolved > 0


def enrich_ratings(df: pd.DataFrame, client: TmdbClient) -> tuple[pd.DataFrame, EnrichmentReport]:
    """Return ``df`` plus TMDB ``genres``/``keywords`` columns and a report.

    ``df`` is expected to be indexed by Const (see ``ratings.load_ratings``).
    Titles missing from TMDB keep their IMDb-export genres so they still get a
    usable feature vector; their ``tmdb_keywords`` stay empty.
    """
    report = EnrichmentReport(total=len(df))
    genres_col: List[List[str]] = []
    keywords_col: List[List[str]] = []
    resolved_col: List[bool] = []
    tmdb_id_col: List[object] = []
    media_type_col: List[object] = []

    for imdb_id, row in df.iterrows():
        fallback_genres = (
            row["genre_list"] if isinstance(row.get("genre_list"), list)
            else split_genres(row.get("Genres"))
        )
        try:
            title = client.fetch_title(str(imdb_id))
        except TmdbError as exc:
            logger.error("TMDB error for %s (%s): %s", imdb_id, row.get("Title"), exc)
            report.errored += 1
            genres_col.append(fallback_genres)
            keywords_col.append([])
            resolved_col.append(False)
            tmdb_id_col.append(pd.NA)
            media_type_col.append(pd.NA)
            continue

        if title is None:
            logger.info("Not on TMDB: %s (%s)", imdb_id, row.get("Title"))
            report.not_found += 1
            genres_col.append(fallback_genres)
            keywords_col.append([])
            resolved_col.append(False)
            tmdb_id_col.append(pd.NA)
            media_type_col.append(pd.NA)
            continue

        report.resolved += 1
        genres_col.append(title.genres or fallback_genres)
        keywords_col.append(title.keywords)
        resolved_col.append(True)
        tmdb_id_col.append(title.tmdb_id)
        media_type_col.append(title.media_type)

    out = df.copy()
    out["genres"] = genres_col
    out["tmdb_keywords"] = keywords_col
    out["tmdb_resolved"] = resolved_col
    out["tmdb_id"] = pd.array(tmdb_id_col, dtype="Int64")
    out["media_type"] = media_type_col

    logger.info(
        "Enrichment: %d total, %d resolved, %d not-found, %d errored",
        report.total, report.resolved, report.not_found, report.errored,
    )
    return out, report
