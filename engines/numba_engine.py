"""
engines/numba_engine.py
-----------------------
Numba/Polars spatial-matching engine (Structure-of-Arrays layout).

Memory model
~~~~~~~~~~~~
Links index — 7 flat contiguous arrays:
  link_starts   float64 (N, 2)  start [lon, lat] of each road link
  link_ends     float64 (N, 2)  end   [lon, lat] of each road link
  sorted_indices int64  (N,)    link row pointers sorted by grid cell key
  sorted_keys    int64  (N,)    corresponding cell keys (for searchsorted)
  min_lon/lat, cell_size, num_cols  — grid parameters

Query batch — one contiguous 2-D array:
  query_coords  float64 (M, 4)  [start_lon, start_lat, end_lon, end_lat]
  (Polars strips WKT strings upfront; Numba never sees a string.)

String metadata (start_node, end_node, roadtype, …) lives only in Polars
and is re-joined from returned integer indices — no strings cross the hot loop.

Contrast with src/lib.rs (Rust engine, Array-of-Structures):
  Vec<LinkData>  — one struct per link, heap-allocated with String fields
  KdTree<f64, usize, [f64;2]>  — pointer-tree spatial index
  WKT parsed by char-splitting inside the rayon parallel loop

Both engines implement the same matching logic:
  1. Candidate pre-filter via spatial index (grid here, KdTree in Rust).
  2. Score each candidate with haversine on forward and reversed orientations.
  3. Keep the link whose combined endpoint distance is minimum and within d_limit.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import numba as nb


# ---------------------------------------------------------------------------
# Numba JIT kernels — compiled once on first call, cached to disk afterward
# ---------------------------------------------------------------------------

@nb.njit(fastmath=True)
def haversine_nb(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres between two lon/lat points."""
    r = 6371000.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c


@nb.njit(parallel=True, fastmath=True)
def match_bulk_numba(
    query_coords: np.ndarray,    # (M, 4) — [start_lon, start_lat, end_lon, end_lat]
    link_starts: np.ndarray,     # (N, 2) — [lon, lat]
    link_ends: np.ndarray,       # (N, 2) — [lon, lat]
    sorted_indices: np.ndarray,  # (N,)   — link row pointers sorted by grid key
    sorted_keys: np.ndarray,     # (N,)   — grid cell keys (for binary search)
    min_lon: float,
    min_lat: float,
    cell_size: float,
    num_cols: int,
    radius_deg: float,
    d_limit_meters: float,
) -> np.ndarray:
    """
    Batch spatial match: each query row → best matching link row index (-1 if none).

    Index structure: flat uniform grid sorted by cell key.  Candidate lookup is
    a pair of np.searchsorted calls instead of tree pointer chasing — the entire
    working set is sequential in memory.  prange parallelises across M queries.
    """
    m = len(query_coords)
    match_indices = np.full(m, -1, dtype=np.int64)
    radius_sq = radius_deg * radius_deg

    for idx in nb.prange(m):
        q_slon = query_coords[idx, 0]
        q_slat = query_coords[idx, 1]
        q_elon = query_coords[idx, 2]
        q_elat = query_coords[idx, 3]

        best_dist = 1e18
        best_link_idx = -1

        # Grid cell of the query's start and end endpoints
        s_c = int((q_slon - min_lon) / cell_size)
        s_r = int((q_slat - min_lat) / cell_size)
        e_c = int((q_elon - min_lon) / cell_size)
        e_r = int((q_elat - min_lat) / cell_size)

        # 3×3 neighbourhood around each endpoint cell to handle cell boundaries
        for dr in range(-1, 2):
            for dc in range(-1, 2):

                # --- neighbourhood around start endpoint ---
                r_s = s_r + dr
                c_s = s_c + dc
                if r_s >= 0 and c_s >= 0 and c_s < num_cols:
                    key = r_s * num_cols + c_s
                    lo = np.searchsorted(sorted_keys, key, side="left")
                    hi = np.searchsorted(sorted_keys, key, side="right")
                    for k in range(lo, hi):
                        l_idx = sorted_indices[k]
                        # cheap squared-degree pre-filter
                        d_sq = ((q_slon - link_starts[l_idx, 0]) ** 2
                                + (q_slat - link_starts[l_idx, 1]) ** 2)
                        if d_sq > radius_sq:
                            continue
                        d_norm_s = haversine_nb(q_slon, q_slat,
                                               link_starts[l_idx, 0], link_starts[l_idx, 1])
                        d_norm_e = haversine_nb(q_elon, q_elat,
                                               link_ends[l_idx, 0], link_ends[l_idx, 1])
                        d_rev_s  = haversine_nb(q_slon, q_slat,
                                               link_ends[l_idx, 0], link_ends[l_idx, 1])
                        d_rev_e  = haversine_nb(q_elon, q_elat,
                                               link_starts[l_idx, 0], link_starts[l_idx, 1])
                        cur = -1.0
                        if d_norm_s <= d_limit_meters and d_norm_e <= d_limit_meters:
                            cur = d_norm_s + d_norm_e
                        elif d_rev_s <= d_limit_meters and d_rev_e <= d_limit_meters:
                            cur = d_rev_s + d_rev_e
                        if 0.0 <= cur < best_dist:
                            best_dist = cur
                            best_link_idx = l_idx

                # --- neighbourhood around end endpoint ---
                r_e = e_r + dr
                c_e = e_c + dc
                if r_e >= 0 and c_e >= 0 and c_e < num_cols:
                    key = r_e * num_cols + c_e
                    lo = np.searchsorted(sorted_keys, key, side="left")
                    hi = np.searchsorted(sorted_keys, key, side="right")
                    for k in range(lo, hi):
                        l_idx = sorted_indices[k]
                        d_sq = ((q_elon - link_starts[l_idx, 0]) ** 2
                                + (q_elat - link_starts[l_idx, 1]) ** 2)
                        if d_sq > radius_sq:
                            continue
                        d_norm_s = haversine_nb(q_slon, q_slat,
                                               link_starts[l_idx, 0], link_starts[l_idx, 1])
                        d_norm_e = haversine_nb(q_elon, q_elat,
                                               link_ends[l_idx, 0], link_ends[l_idx, 1])
                        d_rev_s  = haversine_nb(q_slon, q_slat,
                                               link_ends[l_idx, 0], link_ends[l_idx, 1])
                        d_rev_e  = haversine_nb(q_elon, q_elat,
                                               link_starts[l_idx, 0], link_starts[l_idx, 1])
                        cur = -1.0
                        if d_norm_s <= d_limit_meters and d_norm_e <= d_limit_meters:
                            cur = d_norm_s + d_norm_e
                        elif d_rev_s <= d_limit_meters and d_rev_e <= d_limit_meters:
                            cur = d_rev_s + d_rev_e
                        if 0.0 <= cur < best_dist:
                            best_dist = cur
                            best_link_idx = l_idx

        match_indices[idx] = best_link_idx

    return match_indices


# ---------------------------------------------------------------------------
# Index construction — pure Python + NumPy; runs once at startup
# ---------------------------------------------------------------------------

def build_flat_spatial_index(df_links: pl.DataFrame, cell_size: float) -> dict:
    """
    Build the SoA flat-grid spatial index from a Polars links DataFrame.

    Expected columns: start_lon, start_lat, end_lon, end_lat (all Float64).

    Returns a dict with:
      link_starts    float64 (N, 2)
      link_ends      float64 (N, 2)
      sorted_indices int64   (N,)
      sorted_keys    int64   (N,)
      min_lon, min_lat, cell_size, num_cols
    """
    link_starts = np.ascontiguousarray(
        df_links.select(["start_lon", "start_lat"]).to_numpy(), dtype=np.float64
    )
    link_ends = np.ascontiguousarray(
        df_links.select(["end_lon", "end_lat"]).to_numpy(), dtype=np.float64
    )

    min_lon = link_starts[:, 0].min()
    max_lon = link_starts[:, 0].max()
    min_lat = link_starts[:, 1].min()

    num_cols = int(np.ceil((max_lon - min_lon) / cell_size)) + 1

    c_idx = ((link_starts[:, 0] - min_lon) / cell_size).astype(np.int64)
    r_idx = ((link_starts[:, 1] - min_lat) / cell_size).astype(np.int64)
    cell_keys = r_idx * num_cols + c_idx

    sorted_indices = np.argsort(cell_keys).astype(np.int64)
    sorted_keys = cell_keys[sorted_indices].astype(np.int64)

    return {
        "link_starts": link_starts,
        "link_ends": link_ends,
        "sorted_indices": sorted_indices,
        "sorted_keys": sorted_keys,
        "min_lon": float(min_lon),
        "min_lat": float(min_lat),
        "cell_size": float(cell_size),
        "num_cols": int(num_cols),
    }


def parse_wkt_polars(df_query: pl.DataFrame) -> pl.DataFrame:
    """
    Vectorised WKT → coordinate columns using Polars columnar regex.

    Expects a column named 'wkt' with LINESTRING geometries.
    Returns the input DataFrame augmented with:
      start_lon, start_lat, end_lon, end_lat  (all Float64).

    The heavy string work is done by Polars' compiled regex engine; Numba
    never sees a string.
    """
    return (
        df_query
        .with_columns(
            pl.col("wkt").str.extract_all(r"[-+]?\d*\.\d+|\d+").alias("_coords")
        )
        .with_columns([
            pl.col("_coords").list.get(0).cast(pl.Float64).alias("start_lon"),
            pl.col("_coords").list.get(1).cast(pl.Float64).alias("start_lat"),
            pl.col("_coords").list.get(2).cast(pl.Float64).alias("end_lon"),
            pl.col("_coords").list.get(3).cast(pl.Float64).alias("end_lat"),
        ])
        .drop("_coords")
    )


# ---------------------------------------------------------------------------
# Single-process orchestrator — used by tests and the Dask driver
# ---------------------------------------------------------------------------

def match_dataframe(
    df_links: pl.DataFrame,
    df_query: pl.DataFrame,
    radius_deg: float = 0.0002,
    d_limit_meters: float = 10.0,
    cell_size: float | None = None,
) -> pl.Series:
    """
    Run the full Numba pipeline in a single process.

    Parameters
    ----------
    df_links : pl.DataFrame
        Road-link table with columns start_lon, start_lat, end_lon, end_lat
        plus any string metadata columns (untouched by this function).
    df_query : pl.DataFrame
        Query table with a 'wkt' column (LINESTRING geometries).
    radius_deg : float
        Candidate search radius in degrees (squared-euclidean pre-filter).
    d_limit_meters : float
        Maximum allowable haversine distance for each endpoint to declare a match.
    cell_size : float | None
        Uniform grid cell size in degrees.  Defaults to radius_deg.

    Returns
    -------
    pl.Series  (int64, name "match_idx")
        Row index into df_links for each query row; -1 means no match found.
    """
    if cell_size is None:
        cell_size = radius_deg

    # 1. Parse WKT → coordinate columns (Polars, no Python loops)
    df_parsed = parse_wkt_polars(df_query)

    # 2. Build SoA flat index (NumPy)
    idx = build_flat_spatial_index(df_links, cell_size=cell_size)

    # 3. Extract query coords as a contiguous float64 array (M, 4)
    query_coords = np.ascontiguousarray(
        df_parsed.select(["start_lon", "start_lat", "end_lon", "end_lat"]).to_numpy(),
        dtype=np.float64,
    )

    # 4. Numba parallel kernel
    raw_indices = match_bulk_numba(
        query_coords,
        idx["link_starts"], idx["link_ends"],
        idx["sorted_indices"], idx["sorted_keys"],
        idx["min_lon"], idx["min_lat"],
        idx["cell_size"], idx["num_cols"],
        radius_deg, d_limit_meters,
    )

    return pl.Series("match_idx", raw_indices)
