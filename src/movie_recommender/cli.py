"""Command-line entry point.

    movie-recommender enrich        # fetch TMDB genres+keywords (needs TMDB_API_KEY)
    movie-recommender similarity    # build + save the cosine-similarity matrix
    movie-recommender neighbors --title "Up"     # nearest titles to one you rated
    movie-recommender recommend                  # rating-weighted ranking
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from . import config
from .features import build_features
from .pipeline import load_candidates, load_enriched, run_candidates, run_enrichment
from .recommend import recommend
from .similarity import cosine_similarity_frame, nearest_neighbors
from .tmdb import TmdbClient


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_const(df: pd.DataFrame, title: str) -> str:
    matches = df.index[df["Title"].str.casefold() == title.casefold()]
    if len(matches) == 0:
        raise SystemExit(f"No rated title matches {title!r}.")
    if len(matches) > 1:
        raise SystemExit(f"{title!r} is ambiguous: {list(matches)}")
    return matches[0]


def cmd_enrich(args: argparse.Namespace) -> int:
    client = TmdbClient(config.load_tmdb_config())
    run_enrichment(client, config.RATINGS_CSV, config.ENRICHED_PARQUET)
    return 0


def cmd_similarity(args: argparse.Namespace) -> int:
    df = load_enriched(config.RATINGS_CSV, config.ENRICHED_PARQUET)
    features = build_features(df, genre_weight=args.genre_weight)
    sim = cosine_similarity_frame(features)
    config.SIMILARITY_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    sim.to_parquet(config.SIMILARITY_PARQUET)
    print(f"Wrote {sim.shape[0]}×{sim.shape[1]} similarity matrix to {config.SIMILARITY_PARQUET}")
    return 0


def cmd_neighbors(args: argparse.Namespace) -> int:
    df = load_enriched(config.RATINGS_CSV, config.ENRICHED_PARQUET)
    features = build_features(df, genre_weight=args.genre_weight)
    sim = cosine_similarity_frame(features)
    const = args.const or _resolve_const(df, args.title)
    neighbors = nearest_neighbors(sim, const, top_n=args.top_n)
    titles = df.loc[neighbors.index, "Title"]
    out = neighbors.assign(Title=titles)[["Title", "similarity"]]
    print(f"Nearest to {df.loc[const, 'Title']} ({const}):")
    print(out.to_string())
    return 0


def _parse_csv_set(value: str | None) -> set[str] | None:
    if not value:
        return None
    items = {item.strip().casefold() for item in value.split(",") if item.strip()}
    return items or None


def cmd_candidates(args: argparse.Namespace) -> int:
    enriched = load_enriched(config.RATINGS_CSV, config.ENRICHED_PARQUET)
    client = TmdbClient(config.load_tmdb_config())
    pool = run_candidates(
        client, enriched, config.CANDIDATES_PARQUET,
        min_rating=args.min_rating, max_seeds=args.max_seeds,
        max_candidates=args.max_candidates,
        media_types=_parse_csv_set(args.media_type),
        languages=_parse_csv_set(args.language),
    )
    print(f"Wrote {len(pool)} unwatched candidates to {config.CANDIDATES_PARQUET}")
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    df = load_enriched(config.RATINGS_CSV, config.ENRICHED_PARQUET)
    if "Your Rating" not in df.columns:
        raise SystemExit("Ratings export has no 'Your Rating' column.")

    if args.from_tmdb:
        # Rank a real unwatched pool: put watched + candidates in one feature
        # space, build the taste profile from watched ratings, score candidates.
        candidates = load_candidates(config.CANDIDATES_PARQUET)
        combined = pd.concat([df, candidates])
        features = build_features(combined, genre_weight=args.genre_weight)
        ranked = recommend(
            features, df["Your Rating"],
            candidate_index=candidates.index, top_n=args.top_n,
        )
        meta = candidates.loc[ranked.index]
        cols = [c for c in ("Title", "media_type", "original_language") if c in meta.columns]
        out = ranked.join(meta[cols])[cols + ["score"]]
        header = "Top unwatched recommendations (TMDB candidate pool):"
    else:
        features = build_features(df, genre_weight=args.genre_weight)
        ranked = recommend(features, df["Your Rating"], top_n=args.top_n)
        header = "Top recommendations:"
        if ranked.empty:
            header = (
                "No unwatched candidates: every title in the export is already "
                "rated. Run `candidates` then `recommend --from-tmdb` for real "
                "recommendations.\nShowing your library ranked by taste fit:"
            )
            ranked = recommend(
                features, df["Your Rating"],
                candidate_index=df.index, top_n=args.top_n,
            )
        out = ranked.assign(Title=df.loc[ranked.index, "Title"])[["Title", "score"]]

    print(header)
    print(out.to_string())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, float_format="%.6f")
        print(f"\nWrote {len(out)} recommendations to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--genre-weight", type=float, default=3.0,
                        help="Relative weight of genre vs keyword features (default: 3.0)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("enrich", help="Fetch TMDB genres+keywords").set_defaults(func=cmd_enrich)
    sub.add_parser("similarity", help="Build the cosine-similarity matrix").set_defaults(func=cmd_similarity)

    p_candidates = sub.add_parser("candidates", help="Build an unwatched candidate pool from TMDB")
    p_candidates.add_argument("--min-rating", type=float, default=8.0,
                              help="Only seed from titles rated at least this (default: 8.0)")
    p_candidates.add_argument("--max-seeds", type=int, default=20)
    p_candidates.add_argument("--max-candidates", type=int, default=100)
    p_candidates.add_argument("--media-type", metavar="movie,tv",
                              help="Comma list of media types to keep (default: all)")
    p_candidates.add_argument("--language", metavar="en,ja",
                              help="Comma list of ISO-639-1 original languages to keep (default: all)")
    p_candidates.set_defaults(func=cmd_candidates)

    p_neighbors = sub.add_parser("neighbors", help="Show nearest titles to one you rated")
    g = p_neighbors.add_mutually_exclusive_group(required=True)
    g.add_argument("--const", help="IMDb tt-id")
    g.add_argument("--title", help="Exact rated title")
    p_neighbors.add_argument("--top-n", type=int, default=10)
    p_neighbors.set_defaults(func=cmd_neighbors)

    p_recommend = sub.add_parser("recommend", help="Rating-weighted recommendations")
    p_recommend.add_argument("--top-n", type=int, default=20)
    p_recommend.add_argument("--from-tmdb", action="store_true",
                             help="Rank the unwatched TMDB candidate pool (run `candidates` first)")
    p_recommend.add_argument("--output", type=Path, metavar="PATH",
                             help="Also write the ranking to a CSV file")
    p_recommend.set_defaults(func=cmd_recommend)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
