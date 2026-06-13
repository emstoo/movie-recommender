"""Glue helpers that load/persist intermediate artifacts.

Kept separate from the CLI so the steps can be reused and unit-tested without
going through argparse.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .enrich import enrich_ratings
from .ratings import load_ratings
from .tmdb import TmdbClient

logger = logging.getLogger(__name__)

# List columns are serialized to pipe-joined strings for a stable round-trip.
_LIST_COLUMNS = ("genres", "tmdb_keywords")


def _encode_lists(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in _LIST_COLUMNS:
        if col in out.columns:
            out[col] = out[col].apply(lambda xs: "|".join(xs) if isinstance(xs, list) else "")
    return out


def _decode_lists(df: pd.DataFrame) -> pd.DataFrame:
    for col in _LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").apply(lambda s: [x for x in s.split("|") if x])
    return df


def run_enrichment(client: TmdbClient, ratings_csv: Path, out_path: Path) -> pd.DataFrame:
    df = load_ratings(ratings_csv)
    enriched, report = enrich_ratings(df, client)
    if not report.ok:
        raise RuntimeError(
            "Enrichment resolved zero titles on TMDB — refusing to write an "
            "empty table. Check the TMDB key and network."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _encode_lists(enriched.reset_index()).to_parquet(out_path, index=False)
    logger.info("Wrote enriched table (%d rows) to %s", len(enriched), out_path)
    return enriched


def run_candidates(client: TmdbClient, enriched: pd.DataFrame, out_path: Path,
                   min_rating: float, max_seeds: int, max_candidates: int,
                   media_types=None, languages=None) -> pd.DataFrame:
    from .candidates import build_candidate_pool

    pool = build_candidate_pool(
        enriched, client,
        min_rating=min_rating, max_seeds=max_seeds, max_candidates=max_candidates,
        media_types=media_types, languages=languages,
    )
    if pool.empty:
        raise RuntimeError("TMDB returned no candidates for the selected seeds.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _encode_lists(pool.reset_index()).to_parquet(out_path, index=False)
    logger.info("Wrote candidate pool (%d rows) to %s", len(pool), out_path)
    return pool


def load_candidates(candidates_path: Path) -> pd.DataFrame:
    if not candidates_path.exists():
        raise FileNotFoundError(
            f"No candidate pool at {candidates_path}; run `candidates` first."
        )
    return _decode_lists(pd.read_parquet(candidates_path)).set_index("Const")


def load_enriched(ratings_csv: Path, enriched_path: Path) -> pd.DataFrame:
    """Load the enriched table, or synthesize a genres-only one from ratings.

    This lets ``similarity``/``recommend`` run with no TMDB key (genres only),
    so the pipeline is usable before enrichment has been done.
    """
    if enriched_path.exists():
        df = _decode_lists(pd.read_parquet(enriched_path)).set_index("Const")
        logger.info("Loaded enriched table (%d rows) from %s", len(df), enriched_path)
        return df

    logger.warning(
        "No enriched table at %s; falling back to genres-only features "
        "(run `enrich` with a TMDB key for keyword-based similarity).",
        enriched_path,
    )
    df = load_ratings(ratings_csv)
    df["genres"] = df["genre_list"]
    df["tmdb_keywords"] = [[] for _ in range(len(df))]
    df["tmdb_resolved"] = False
    return df
