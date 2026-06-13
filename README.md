# movie-recommender

*English | [日本語](README.ja.md)*

Content-based movie recommender built from an IMDb "Your Ratings" CSV export,
using TMDB-derived genres + keywords as features.

It starts from your own IMDb ratings, builds a taste profile from how you
scored them, and ranks an unwatched candidate pool pulled live from the TMDB
API. It runs as a small CLI pipeline.

## Setup

```bash
uv sync
# Use ONE of these (token is preferred). Both come from
# https://www.themoviedb.org/settings/api
export TMDB_API_TOKEN=eyJ...     # v4 "API Read Access Token" (Authorization: Bearer)
# export TMDB_API_KEY=xxxxxxxx   # v3 API key (api_key query param)
```

Export your ratings from IMDb (Your Ratings → ⋯ → Export) and save the CSV to
`data/raw/movies.csv`. The `data/` directory is git-ignored — your ratings stay
local and are never committed.

## Pipeline

```bash
uv run movie-recommender enrich                   # ① TMDB → genres + keywords (needs credential)
uv run movie-recommender similarity               # ② cosine-similarity matrix
uv run movie-recommender neighbors --title "Up"   #    nearest rated titles to one you rated
uv run movie-recommender candidates               # ③ unwatched candidate pool from TMDB
uv run movie-recommender recommend --from-tmdb     # ④ rank that pool by your taste
```

`recommend` without `--from-tmdb` (and `similarity`/`neighbors`) also run with
no TMDB credential, falling back to genres-only features — lower quality, no
network. `candidates` and `recommend --from-tmdb` require a credential.

### Artifacts

| Path | Written by |
|------|------------|
| `data/interim/enriched.parquet` | `enrich` |
| `data/interim/similarity.parquet` | `similarity` |
| `data/interim/candidates.parquet` | `candidates` |
| CSV via `recommend --output PATH` | `recommend` (stdout-only by default) |

## Architecture

| Module | Responsibility |
|--------|----------------|
| `ratings.py` | Load + normalize the IMDb export (Const-keyed) |
| `tmdb.py` | TMDB API client (loud failures, retries; movies + TV) |
| `enrich.py` | Attach TMDB genres/keywords; report resolution stats |
| `features.py` | Genres+keywords → L2-normalized sparse feature matrix |
| `similarity.py` | Pairwise cosine similarity (single matrix, Const-keyed) |
| `candidates.py` | Build an unwatched candidate pool from TMDB recommendations |
| `recommend.py` | Rating-weighted taste profile → candidate ranking |
| `pipeline.py` | Load/persist intermediate artifacts |
| `cli.py` | `movie-recommender` entry point |

## Tuning

- `--genre-weight` (default 3.0): how much genres count vs keywords when building
  features. Global flag used by `similarity`, `neighbors`, and `recommend` —
  it goes **before** the subcommand, e.g.
  `movie-recommender --genre-weight 2.0 recommend --from-tmdb`.
- `candidates --min-rating / --max-seeds / --max-candidates`: which favourites
  seed the pool and how large it gets.
- `candidates --media-type movie,tv`: keep only these media types (default: all).
- `candidates --language ja,en`: keep only these ISO-639-1 original languages
  (default: all). Both filters apply before feature fetches, so excluded
  candidates cost no API calls.
- `recommend --output PATH`: also write the ranking to CSV (default: stdout only).
- `recommend --top-n N`: how many titles to return.

```bash
# e.g. only Japanese-language movies, saved to CSV
uv run movie-recommender candidates --media-type movie --language ja
uv run movie-recommender recommend --from-tmdb --output data/processed/recommendations.csv
```

## How it works

- **Const (tt-id) is the key** throughout the pipeline, removing the old
  dependency on filenames built from title + year.
- Distances use the cosine similarity of L2-normalized vectors in [0, 1]
  directly — **no per-row min-max normalization** — so scores are comparable
  across titles.
- Recommendations turn `Your Rating` into a taste profile using
  **mean-centered weights**: titles rated above your average pull the profile
  toward their content, those below push it away. A negative score means a
  candidate resembles titles you rated below your average — i.e. relatively off
  your taste.
- IMDb scraping (cinemagoer) no longer works, so the feature source is the
  TMDB API.

## Possible next steps

- De-duplicate franchises / sequels in the candidate pool.
- Blend in TMDB popularity/vote_average so niche-but-similar titles don't
  always outrank broadly loved ones.
- Persist the taste profile and add an offline evaluation (leave-one-out).

## License

[MIT](LICENSE). This product uses the TMDB API but is not endorsed or certified
by TMDB.
