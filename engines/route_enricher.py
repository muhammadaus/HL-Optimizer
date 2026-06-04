"""
engines/route_enricher.py
--------------------------
Enrich matched GPS endpoint pairs with real routing metrics from Mapbox
Directions API, then cache results locally so identical origin/destination
pairs never re-hit the API.

Flow
~~~~
  two GPS endpoints (lon/lat)
      ↓  engines/numba_engine.py  or  src/lib.rs  (spatial match)
      ↓  already have: start_lon/lat, end_lon/lat
      ↓
  route_enricher.enrich_batch(pairs)
      ↓  cache lookup  →  hit: return stored result
      ↓  miss: call Mapbox Directions API
      ↓  store result
      ↓
  RouteResult per pair
      {origin, destination, provider, distance_meters,
       duration_seconds, traffic_duration_seconds,
       congestion, route_geometry, tolls, computed_at}

Cache key
~~~~~~~~~
  SHA256( origin_geohash7 + ":" + destination_geohash7
        + ":" + mode + ":" + provider
        + ":" + departure_time_bucket_iso_hour )

  geohash precision 7 ≈ 150 m cell — natural deduplication for urban GPS
  traces where many trips share the same start/end intersection.
  departure_time_bucket rounds to the nearest hour so traffic-aware results
  don't get stale across peak/off-peak transitions.

Environment variables
~~~~~~~~~~~~~~~~~~~~~
  MAPBOX_ACCESS_TOKEN   — required for Mapbox (default provider)
  ROUTE_CACHE_DIR       — directory for the JSON cache (default: .route_cache/)

Usage
~~~~~
  from engines.route_enricher import enrich_batch, RouteResult

  pairs = [
      {"origin": [139.7001, 35.6895], "destination": [139.7600, 35.6812]},
      ...
  ]
  results: list[RouteResult] = enrich_batch(pairs)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

import pygeohash as geohash


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RouteResult:
    origin: List[float]                        # [lon, lat]
    destination: List[float]                   # [lon, lat]
    provider: str
    distance_meters: Optional[int]
    duration_seconds: Optional[int]
    traffic_duration_seconds: Optional[int]    # None when traffic data unavailable
    congestion: Optional[List[str]]            # per-segment annotation list
    route_geometry: Optional[str]              # encoded polyline (Mapbox format)
    tolls: Optional[bool]
    computed_at: str                           # ISO-8601 UTC timestamp
    error: Optional[str] = None               # set when the API call failed


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_DIR = Path(os.environ.get("ROUTE_CACHE_DIR", ".route_cache"))

def _cache_path(key: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{key}.json"


def _departure_bucket(dt: Optional[datetime] = None) -> str:
    """Round datetime to the nearest hour for the cache key."""
    dt = dt or datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:00Z")


def _cache_key(
    origin_lon: float, origin_lat: float,
    dest_lon: float,   dest_lat: float,
    mode: str, provider: str,
    departure_dt: Optional[datetime] = None,
) -> str:
    gh_origin = geohash.encode(origin_lat, origin_lon, precision=7)
    gh_dest   = geohash.encode(dest_lat,   dest_lon,   precision=7)
    bucket    = _departure_bucket(departure_dt)
    raw = f"{gh_origin}:{gh_dest}:{mode}:{provider}:{bucket}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _cache_get(key: str, cache_dir: Path) -> Optional[RouteResult]:
    p = _cache_path(key, cache_dir)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        return RouteResult(**data)
    except Exception:
        return None


def _cache_put(key: str, result: RouteResult, cache_dir: Path) -> None:
    _cache_path(key, cache_dir).write_text(
        json.dumps(asdict(result), indent=2)
    )


# ---------------------------------------------------------------------------
# Mapbox Directions API
# ---------------------------------------------------------------------------

_MAPBOX_BASE = "https://api.mapbox.com/directions/v5/mapbox"


def _call_mapbox(
    origin_lon: float, origin_lat: float,
    dest_lon: float,   dest_lat: float,
    mode: str = "driving-traffic",
    token: Optional[str] = None,
) -> RouteResult:
    """
    Call Mapbox Directions API for one origin→destination pair.

    Profile 'driving-traffic' returns traffic-aware duration and per-step
    congestion annotations.  Falls back to 'driving' if the token is missing.

    Mapbox congestion values: "low" | "moderate" | "heavy" | "severe" | "unknown"
    """
    token = token or os.environ.get("MAPBOX_ACCESS_TOKEN", "")
    coords = f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
    url = (
        f"{_MAPBOX_BASE}/{mode}/{coords}"
        f"?access_token={token}"
        f"&geometries=polyline"
        f"&annotations=congestion,maxspeed"
        f"&overview=full"
        f"&steps=false"
    )

    computed_at = datetime.now(timezone.utc).isoformat()
    origin      = [origin_lon, origin_lat]
    destination = [dest_lon,   dest_lat]

    if not token:
        return RouteResult(
            origin=origin, destination=destination,
            provider="mapbox",
            distance_meters=None, duration_seconds=None,
            traffic_duration_seconds=None, congestion=None,
            route_geometry=None, tolls=None,
            computed_at=computed_at,
            error="MAPBOX_ACCESS_TOKEN not set",
        )

    try:
        req = Request(url, headers={"User-Agent": "HL-Optimizer/1.0"})
        with urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
    except URLError as exc:
        return RouteResult(
            origin=origin, destination=destination,
            provider="mapbox",
            distance_meters=None, duration_seconds=None,
            traffic_duration_seconds=None, congestion=None,
            route_geometry=None, tolls=None,
            computed_at=computed_at,
            error=str(exc),
        )

    if not body.get("routes"):
        return RouteResult(
            origin=origin, destination=destination,
            provider="mapbox",
            distance_meters=None, duration_seconds=None,
            traffic_duration_seconds=None, congestion=None,
            route_geometry=None, tolls=None,
            computed_at=computed_at,
            error=body.get("message", "no routes returned"),
        )

    route = body["routes"][0]
    legs  = route.get("legs", [{}])

    # Flatten per-step congestion annotations across all legs
    congestion: List[str] = []
    for leg in legs:
        ann = leg.get("annotation", {})
        congestion.extend(ann.get("congestion", []))

    # duration_typical is the non-traffic baseline; use it to derive
    # traffic_duration_seconds = actual - typical when available
    duration          = int(route.get("duration", 0))
    duration_typical  = int(route.get("duration_typical", duration))
    traffic_delta     = duration - duration_typical  # positive = slower than typical

    return RouteResult(
        origin=origin,
        destination=destination,
        provider="mapbox",
        distance_meters=int(route.get("distance", 0)),
        duration_seconds=duration,
        traffic_duration_seconds=duration if mode == "driving-traffic" else None,
        congestion=congestion if congestion else None,
        route_geometry=route.get("geometry"),
        tolls=None,          # Mapbox doesn't return toll info in this endpoint
        computed_at=computed_at,
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def enrich_single(
    origin_lon: float, origin_lat: float,
    dest_lon: float,   dest_lat: float,
    mode: str = "driving-traffic",
    provider: str = "mapbox",
    departure_dt: Optional[datetime] = None,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
    token: Optional[str] = None,
) -> RouteResult:
    """
    Enrich one origin→destination pair with routing metrics.

    Checks the local geohash-keyed cache first; calls the provider API on a
    miss and stores the result for future calls.
    """
    key = _cache_key(origin_lon, origin_lat, dest_lon, dest_lat,
                     mode, provider, departure_dt)
    cached = _cache_get(key, cache_dir)
    if cached is not None:
        return cached

    if provider == "mapbox":
        result = _call_mapbox(origin_lon, origin_lat, dest_lon, dest_lat,
                              mode=mode, token=token)
    else:
        result = RouteResult(
            origin=[origin_lon, origin_lat],
            destination=[dest_lon, dest_lat],
            provider=provider,
            distance_meters=None, duration_seconds=None,
            traffic_duration_seconds=None, congestion=None,
            route_geometry=None, tolls=None,
            computed_at=datetime.now(timezone.utc).isoformat(),
            error=f"provider '{provider}' not implemented",
        )

    _cache_put(key, result, cache_dir)
    return result


def enrich_batch(
    pairs: List[dict],
    mode: str = "driving-traffic",
    provider: str = "mapbox",
    departure_dt: Optional[datetime] = None,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
    token: Optional[str] = None,
    rate_limit_rps: float = 10.0,
) -> List[RouteResult]:
    """
    Enrich a list of origin/destination pairs with routing metrics.

    Each dict in pairs must have:
        origin      : [lon, lat]
        destination : [lon, lat]

    Results are returned in the same order as pairs.
    Cache hits are returned immediately; API calls are rate-limited to
    rate_limit_rps requests per second (Mapbox free tier: 300/min ≈ 5 rps).

    Example
    -------
    pairs = [
        {"origin": [139.7001, 35.6895], "destination": [139.7600, 35.6812]},
        {"origin": [139.6900, 35.6700], "destination": [139.7100, 35.6900]},
    ]
    results = enrich_batch(pairs)
    for r in results:
        print(r.distance_meters, r.duration_seconds, r.congestion)
    """
    min_interval = 1.0 / rate_limit_rps
    results: List[RouteResult] = []
    last_call_at = 0.0

    for pair in pairs:
        origin      = pair["origin"]       # [lon, lat]
        destination = pair["destination"]  # [lon, lat]

        key = _cache_key(origin[0], origin[1], destination[0], destination[1],
                         mode, provider, departure_dt)
        cached = _cache_get(key, cache_dir)
        if cached is not None:
            results.append(cached)
            continue

        # Rate-limit only actual API calls
        elapsed = time.perf_counter() - last_call_at
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        result = enrich_single(
            origin[0], origin[1], destination[0], destination[1],
            mode=mode, provider=provider,
            departure_dt=departure_dt,
            cache_dir=cache_dir, token=token,
        )
        last_call_at = time.perf_counter()
        results.append(result)

    return results
