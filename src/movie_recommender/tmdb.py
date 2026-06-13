"""Thin TMDB API client.

Design principles (learned from the broken cinemagoer pipeline):
    - Network/HTTP failures are surfaced, not swallowed. Callers decide policy.
    - Retries cover only transient errors (timeouts, 429, 5xx); a 404 means the
      title genuinely is not on TMDB and is returned as ``None`` immediately.
    - Both movies and TV series are supported, since an IMDb export mixes them.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from .config import TmdbConfig

logger = logging.getLogger(__name__)

# Media kinds TMDB distinguishes; we map IMDb "Title Type" onto these.
MOVIE = "movie"
TV = "tv"


@dataclass
class TmdbTitle:
    """Resolved TMDB content for one IMDb title."""

    imdb_id: str
    tmdb_id: int
    media_type: str
    genres: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


class TmdbError(RuntimeError):
    """Raised when a TMDB request fails after exhausting retries."""


class TmdbClient:
    def __init__(self, config: TmdbConfig, session: Optional[requests.Session] = None):
        self._config = config
        self._session = session or requests.Session()
        if config.bearer_token:
            # v4 auth: token goes in the header, nothing in the query string.
            self._session.headers["Authorization"] = f"Bearer {config.bearer_token}"

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Optional[dict]:
        """GET ``path`` with retries. Returns parsed JSON, or None on 404."""
        url = f"{self._config.base_url}{path}"
        query = dict(params or {})
        if self._config.api_key:
            # v3 auth: key as a query param (header may already carry a bearer).
            query["api_key"] = self._config.api_key

        last_exc: Optional[Exception] = None
        for attempt in range(1, self._config.max_retries + 1):
            try:
                resp = self._session.get(
                    url, params=query, timeout=self._config.request_timeout
                )
                if resp.status_code == 404:
                    return None
                if resp.status_code == 429:
                    # Respect TMDB's rate-limit hint when present.
                    wait = float(resp.headers.get("Retry-After", self._config.backoff_seconds))
                    logger.warning("TMDB rate-limited; sleeping %.1fs", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                logger.warning("TMDB transient error (attempt %d): %s", attempt, exc)
                time.sleep(self._config.backoff_seconds * attempt)
            except requests.HTTPError as exc:
                # 5xx is worth retrying; other 4xx are not.
                status = exc.response.status_code if exc.response is not None else 0
                if 500 <= status < 600 and attempt < self._config.max_retries:
                    last_exc = exc
                    time.sleep(self._config.backoff_seconds * attempt)
                    continue
                raise TmdbError(f"GET {path} failed: {exc}") from exc

        raise TmdbError(f"GET {path} failed after {self._config.max_retries} retries: {last_exc}")

    def find_by_imdb_id(self, imdb_id: str) -> Optional[tuple[int, str]]:
        """Resolve a tt-id to ``(tmdb_id, media_type)`` via TMDB's /find."""
        data = self._get(f"/find/{imdb_id}", {"external_source": "imdb_id"})
        if not data:
            return None
        if data.get("movie_results"):
            return data["movie_results"][0]["id"], MOVIE
        if data.get("tv_results"):
            return data["tv_results"][0]["id"], TV
        return None

    def fetch_features(
        self, tmdb_id: int, media_type: str = MOVIE
    ) -> tuple[List[str], List[str], Optional[str]]:
        """Return ``(genres, keywords, imdb_id)`` for a TMDB id (movie or TV).

        ``imdb_id`` comes free from the movie detail payload; TV details lack it,
        so we spend one extra ``/external_ids`` call. ``None`` when TMDB has no
        IMDb mapping for the title.
        """
        detail = self._get(f"/{media_type}/{tmdb_id}") or {}
        genres = [g["name"] for g in detail.get("genres", []) if g.get("name")]

        kw_data = self._get(f"/{media_type}/{tmdb_id}/keywords") or {}
        # Movies return "keywords"; TV returns "results". Same shape otherwise.
        raw_keywords = kw_data.get("keywords") or kw_data.get("results") or []
        keywords = [k["name"] for k in raw_keywords if k.get("name")]

        imdb_id = detail.get("imdb_id")  # present for movies
        if not imdb_id and media_type == TV:
            ext = self._get(f"/{media_type}/{tmdb_id}/external_ids") or {}
            imdb_id = ext.get("imdb_id")
        return genres, keywords, (imdb_id or None)

    def fetch_title(self, imdb_id: str) -> Optional[TmdbTitle]:
        """Fetch genres + keywords for one IMDb title. None if not on TMDB."""
        found = self.find_by_imdb_id(imdb_id)
        if found is None:
            return None
        tmdb_id, media_type = found
        genres, keywords, _ = self.fetch_features(tmdb_id, media_type)
        return TmdbTitle(
            imdb_id=imdb_id,
            tmdb_id=tmdb_id,
            media_type=media_type,
            genres=genres,
            keywords=keywords,
        )

    def recommendations(self, tmdb_id: int, media_type: str = MOVIE) -> List[dict]:
        """Return TMDB's recommended titles for a given title (candidate pool)."""
        data = self._get(f"/{media_type}/{tmdb_id}/recommendations") or {}
        return data.get("results", [])
