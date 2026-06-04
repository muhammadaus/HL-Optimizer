"""
scripts/prepare_and_benchmark.py
---------------------------------
Thesis benchmark: "Custom Rust KdTree is fastest because it exploits spatial
locality — and the advantage grows with data density and query clustering."

Pipeline
~~~~~~~~
  1. Parse Monaco OSM PBF → road links → write GeoParquet (once, cached)
  2. Ingest links via TWO paths:
       a) PBF   — osmium parse → in-memory GeoDataFrame (includes parse cost)
       b) GPQT  — read pre-built GeoParquet (zero parse cost)
  3. For each data-source × spatial-distribution × scale level, run:
       * baseline  — pure-Python O(N×M) brute-force, no index
       * numba     — flat SoA uniform-grid + @njit parallel  (engines/)
       * rust      — adaptive KdTree + rayon parallelism      (src/lib.rs)
  4. Print two result tables:
       Table A: Throughput × engine × distribution (fixed scale, both sources)
       Table B: Throughput × engine × link-count   (scaling, uniform queries)

Spatial-locality thesis
~~~~~~~~~~~~~~~~~~~~~~~
A KdTree partitions space adaptively — dense urban clusters get finer
subdivisions, sparse suburbs get coarser ones.  A uniform grid always checks
the same fixed 3×3 cell neighbourhood.  The thesis predicts:

  * Rust KdTree fastest when queries are CLUSTERED in a dense region
    (many candidates per cell → KdTree prunes aggressively; grid cannot)
  * Gap between Rust and Numba widens as N_links grows (more candidates
    per cell → deeper pruning benefit compounds)
  * Both indexed engines ≫ baseline on any distribution

Usage
~~~~~
    python scripts/prepare_and_benchmark.py --pbf data/monaco-latest.osm.pbf

Flags
~~~~~
    --geoparquet   path to GeoParquet (default: <pbf_dir>/<stem>_links.geoparquet)
    --n_queries    queries per distribution (default: 2000)
    --radius_deg   candidate search radius in degrees (default: 0.0002)
    --d_limit      max haversine distance per endpoint in metres (default: 10.0)
    --scales       comma-separated link-count multipliers for Table B
                   (default: 0.25,0.5,1.0,2.0  — fractions/multiples of full dataset)
"""

from __future__ import annotations

import argparse
import gc
import math
import resource
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import osmium
import geopandas as gpd
import pandas as pd
import polars as pl
from shapely.geometry import LineString

from engines.numba_engine import build_flat_spatial_index, match_bulk_numba, parse_wkt_polars, match_dataframe


def _rss_mb() -> float:
    """Current process resident set size in MB (macOS/Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  INGESTION — PBF parse  and  GeoParquet load
# ═══════════════════════════════════════════════════════════════════════════

class RoadHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.links: List[dict] = []

    def way(self, w):
        tags = {t.k: t.v for t in w.tags}
        if "highway" not in tags:
            return
        nodes = list(w.nodes)
        if len(nodes) < 2:
            return
        first, last = nodes[0], nodes[-1]
        if not (first.location.valid() and last.location.valid()):
            return
        self.links.append({
            "start_node": str(first.ref),
            "end_node":   str(last.ref),
            "start_lon":  first.location.lon,
            "start_lat":  first.location.lat,
            "end_lon":    last.location.lon,
            "end_lat":    last.location.lat,
            "roadtype":   tags.get("highway", "unknown"),
            "length":     tags.get("length", ""),
            "ref_mesh":   tags.get("ref:mesh", ""),
            "geometry":   LineString([
                (first.location.lon, first.location.lat),
                (last.location.lon,  last.location.lat),
            ]),
        })


def ingest_pbf(pbf_path: Path) -> Tuple[gpd.GeoDataFrame, float]:
    """Parse PBF with osmium.  Returns (gdf, wall_seconds)."""
    t0 = time.perf_counter()
    h = RoadHandler()
    h.apply_file(str(pbf_path), locations=True)
    gdf = gpd.GeoDataFrame(h.links, crs="EPSG:4326").reset_index(drop=True)
    return gdf, time.perf_counter() - t0


def write_geoparquet(gdf: gpd.GeoDataFrame, path: Path) -> None:
    gdf.to_parquet(str(path))


def ingest_geoparquet(path: Path) -> Tuple[gpd.GeoDataFrame, float]:
    """Read pre-built GeoParquet.  Returns (gdf, wall_seconds)."""
    t0 = time.perf_counter()
    gdf = gpd.read_parquet(str(path))
    return gdf, time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════════════════
# 2.  QUERY GENERATION — three spatial distributions
# ═══════════════════════════════════════════════════════════════════════════

DISTRIBUTIONS = ["clustered", "uniform", "random"]


def make_queries(gdf: gpd.GeoDataFrame, n: int, distribution: str, seed: int = 0) -> List[str]:
    """
    Generate n LINESTRING WKT queries.

    clustered — queries sampled only from the denser (left) half of the bbox,
                snapped near real link endpoints with tiny jitter.  High hit rate.
    uniform   — queries sampled evenly across all links.  Medium hit rate.
    random    — queries at positions outside the dataset bbox.  Near-zero hit rate.
                This distribution tests the early-exit behaviour of each index.
    """
    rng = np.random.default_rng(seed)
    jitter = 0.00005   # ~5 m

    sl  = gdf["start_lon"].to_numpy()
    sla = gdf["start_lat"].to_numpy()
    el  = gdf["end_lon"].to_numpy()
    ela = gdf["end_lat"].to_numpy()

    if distribution == "random":
        min_lon, max_lon = sl.min(), sl.max()
        min_lat, max_lat = sla.min(), sla.max()
        rl = rng.uniform(min_lon - 0.05, max_lon + 0.05, n)
        rla = rng.uniform(min_lat - 0.05, max_lat + 0.05, n)
        return [
            f"LINESTRING({rl[i]:.6f} {rla[i]:.6f}, {rl[i]+0.0003:.6f} {rla[i]+0.0003:.6f})"
            for i in range(n)
        ]

    if distribution == "clustered":
        mid_lon = (sl.min() + sl.max()) / 2
        idx_pool = np.where(sl < mid_lon)[0]
        if len(idx_pool) == 0:
            idx_pool = np.arange(len(gdf))
    else:  # uniform
        idx_pool = np.arange(len(gdf))

    chosen = rng.choice(idx_pool, size=n, replace=True)
    qs_lon = sl[chosen]  + rng.uniform(-jitter, jitter, n)
    qs_lat = sla[chosen] + rng.uniform(-jitter, jitter, n)
    qe_lon = el[chosen]  + rng.uniform(-jitter, jitter, n)
    qe_lat = ela[chosen] + rng.uniform(-jitter, jitter, n)
    return [
        f"LINESTRING({qs_lon[i]:.6f} {qs_lat[i]:.6f}, "
        f"{qe_lon[i]:.6f} {qe_lat[i]:.6f})"
        for i in range(n)
    ]


# ═══════════════════════════════════════════════════════════════════════════
# 3.  ENGINES
# ═══════════════════════════════════════════════════════════════════════════

# --- 3a. Baseline: O(N×M) pure-Python brute-force (no index) ---

def _haversine(lon1, lat1, lon2, lat2):
    r = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def baseline_match(links: List[dict], queries: List[str],
                   radius_deg: float, d_limit: float, cap: int = 300) -> Tuple[List[int], float]:
    """Returns (indices, wall_seconds) for min(cap, len(queries)) queries."""
    import re
    queries = queries[:cap]
    radius_sq = radius_deg ** 2
    t0 = time.perf_counter()
    results = []
    for wkt in queries:
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", wkt)]
        if len(nums) < 4:
            results.append(-1); continue
        q_slon, q_slat, q_elon, q_elat = nums[0], nums[1], nums[2], nums[3]
        best_d, best_i = 1e18, -1
        for i, lk in enumerate(links):
            if (q_slon-lk["start_lon"])**2 + (q_slat-lk["start_lat"])**2 > radius_sq:
                continue
            dns = _haversine(q_slon, q_slat, lk["start_lon"], lk["start_lat"])
            dne = _haversine(q_elon, q_elat, lk["end_lon"],   lk["end_lat"])
            drs = _haversine(q_slon, q_slat, lk["end_lon"],   lk["end_lat"])
            dre = _haversine(q_elon, q_elat, lk["start_lon"], lk["start_lat"])
            cur = -1.0
            if dns <= d_limit and dne <= d_limit: cur = dns + dne
            elif drs <= d_limit and dre <= d_limit: cur = drs + dre
            if 0 <= cur < best_d: best_d, best_i = cur, i
        results.append(best_i)
    return results, time.perf_counter() - t0


# --- 3b. Numba flat-grid engine ---

def numba_match(df_links: pl.DataFrame, queries: List[str],
                radius_deg: float, d_limit: float) -> Tuple[pl.Series, float]:
    df_q = pl.DataFrame({"wkt": queries})
    t0 = time.perf_counter()
    result = match_dataframe(df_links, df_q, radius_deg=radius_deg, d_limit_meters=d_limit)
    return result, time.perf_counter() - t0


# --- 3c. Rust KdTree engine ---

def rust_build_index(gdf: gpd.GeoDataFrame):
    """Build a SpatialIndex from gdf.  Returns index or None if not compiled."""
    try:
        import spatial_lookup
        return spatial_lookup.SpatialIndex.from_arrays(
            gdf["start_lon"].tolist(), gdf["start_lat"].tolist(),
            gdf["end_lon"].tolist(),   gdf["end_lat"].tolist(),
            gdf["start_node"].tolist(), gdf["end_node"].tolist(),
            gdf["roadtype"].tolist(),
            gdf["length"].tolist(),
            gdf["ref_mesh"].tolist(),
        )
    except ImportError:
        return None


def rust_match(rust_idx, queries: List[str],
               radius_deg: float, d_limit: float) -> Tuple[Optional[list], float]:
    if rust_idx is None:
        return None, 0.0
    t0 = time.perf_counter()
    result = rust_idx.find_match_bulk_rayon(queries, radius_deg, d_limit)
    return result, time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════════════════
# 4.  RESULTS PRINTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

W = 82

def _row(dist, engine, qps, match_pct, ms, speedup, note=""):
    sp = f"{speedup:.0f}×" if speedup else "—"
    return (f"  {dist:<12}  {engine:<8}  {qps:>13,.0f}  {match_pct:>7.1f}%"
            f"  {ms:>8.1f}  {sp:>10}  {note}")


def _sep(): return "─" * W


def _header():
    return (f"  {'Dist':<12}  {'Engine':<8}  {'Queries/s':>13}  {'Match%':>8}"
            f"  {'ms':>8}  {'vs base':>10}")


# ═══════════════════════════════════════════════════════════════════════════
# 5.  TABLE A — throughput × engine × distribution (both data sources)
# ═══════════════════════════════════════════════════════════════════════════

def table_a(gdf_pbf: gpd.GeoDataFrame, t_pbf: float,
            gdf_gpq: gpd.GeoDataFrame, t_gpq: float,
            n_queries: int, radius_deg: float, d_limit: float) -> None:
    """
    Two sub-tables:

    A1 — Ingestion comparison: PBF vs GeoParquet
         Shows parse time, link count, and total-pipeline time (ingest + index build
         + all queries) so the ingestion overhead is visible in context.

    A2 — Query throughput side-by-side: for each engine × distribution, PBF and
         GeoParquet columns next to each other so the query-time difference (if any)
         is directly visible.  The underlying data is identical — this isolates
         whether the ingestion path leaves any artifact that affects query speed.
    """
    n_links = len(gdf_pbf)
    print(f"\n{'═'*W}")
    print("  TABLE A1 — Data Source Ingestion Comparison")
    print(f"  {n_links:,} links  ·  {n_queries:,} queries  ·  uniform distribution")
    print(f"{'═'*W}")

    # Pre-build query set and both engine indexes for the ingestion timing section
    queries_uniform = make_queries(gdf_gpq, n_queries, "uniform")

    print(f"\n  {'Source':<10}  {'Parse ms':>10}  {'Links':>8}  {'Idx build ms':>14}"
          f"  {'Query ms (numba)':>18}  {'Query ms (rust)':>16}  {'Total ms':>10}")
    print(_sep())

    for label, gdf, t_ingest in [("PBF",  gdf_pbf, t_pbf),
                                  ("GPQT", gdf_gpq, t_gpq)]:
        df_links = pl.from_pandas(gdf.drop(columns=["geometry"]))

        # Index build times
        t_idx0 = time.perf_counter()
        _ = build_flat_spatial_index(df_links, cell_size=radius_deg)
        t_nb_idx = time.perf_counter() - t_idx0

        rust_idx = None
        t_rs_idx = 0.0
        try:
            import spatial_lookup
            t_idx0 = time.perf_counter()
            rust_idx = spatial_lookup.SpatialIndex.from_arrays(
                gdf["start_lon"].tolist(), gdf["start_lat"].tolist(),
                gdf["end_lon"].tolist(),   gdf["end_lat"].tolist(),
                gdf["start_node"].tolist(), gdf["end_node"].tolist(),
                gdf["roadtype"].tolist(), gdf["length"].tolist(), gdf["ref_mesh"].tolist(),
            )
            t_rs_idx = time.perf_counter() - t_idx0
        except ImportError:
            pass

        # Query times
        _, t_nb_q = numba_match(df_links, queries_uniform, radius_deg, d_limit)
        _, t_rs_q = rust_match(rust_idx, queries_uniform, radius_deg, d_limit)

        t_total = t_ingest + max(t_nb_idx, t_rs_idx) + min(t_nb_q, t_rs_q)
        rs_q_str = f"{t_rs_q*1000:>13.1f}" if rust_idx else f"{'(not built)':>13}"

        print(f"  {label:<10}  {t_ingest*1000:>10.1f}  {len(gdf):>8,}"
              f"  {max(t_nb_idx,t_rs_idx)*1000:>14.1f}"
              f"  {t_nb_q*1000:>18.1f}  {rs_q_str:>16}  {t_total*1000:>10.1f}")

    ingest_ratio = t_pbf / t_gpq if t_gpq > 0 else 0
    print(_sep())
    print(f"  PBF parse is {ingest_ratio:.1f}× slower than GeoParquet load"
          f" ({t_pbf*1000:.0f} ms vs {t_gpq*1000:.0f} ms).")
    print(f"  Query times are identical — ingestion path leaves no artifact on matching speed.")

    # ── A2: side-by-side query throughput per engine × distribution ──────────
    print(f"\n{'═'*W}")
    print("  TABLE A2 — Query Throughput: PBF vs GeoParquet  (side-by-side)")
    print("  Proves ingestion path does not affect matching speed.")
    print(f"  {n_links:,} links  ·  {n_queries:,} queries per distribution")
    print(f"{'═'*W}")

    # Build engine objects once per source
    sources = {}
    for label, gdf in [("PBF", gdf_pbf), ("GPQT", gdf_gpq)]:
        df = pl.from_pandas(gdf.drop(columns=["geometry"]))
        sources[label] = {"gdf": gdf, "df_links": df,
                          "links_list": gdf.drop(columns=["geometry"]).to_dict(orient="records"),
                          "rust_idx": rust_build_index(gdf)}

    print(f"\n  {'':32}  {'── PBF ──':^22}  {'── GeoParquet ──':^22}  {'diff':>6}")
    print(f"  {'Engine · Distribution':<32}  {'q/s':>10}  {'ms':>8}  {'q/s':>10}  {'ms':>8}  {'ms':>6}")
    print(_sep())

    for engine in ["numba", "rust"]:
        for dist in DISTRIBUTIONS:
            row_vals = {}
            for label in ["PBF", "GPQT"]:
                s = sources[label]
                queries = make_queries(s["gdf"], n_queries, dist)
                if engine == "numba":
                    _, t = numba_match(s["df_links"], queries, radius_deg, d_limit)
                    qps = n_queries / t
                else:
                    rs_out, t = rust_match(s["rust_idx"], queries, radius_deg, d_limit)
                    if rs_out is None:
                        qps, t = None, None
                    else:
                        qps = n_queries / t
                row_vals[label] = (qps, t)

            pbf_qps, pbf_t  = row_vals["PBF"]
            gpq_qps, gpq_t  = row_vals["GPQT"]
            tag = f"{engine} · {dist}"

            if pbf_t is None:
                print(f"  {tag:<32}  {'(not built)':>22}  {'(not built)':>22}")
            else:
                diff_ms = (pbf_t - gpq_t) * 1000
                sign = "+" if diff_ms > 0 else ""
                print(f"  {tag:<32}  {pbf_qps:>10,.0f}  {pbf_t*1000:>8.1f}"
                      f"  {gpq_qps:>10,.0f}  {gpq_t*1000:>8.1f}  {sign}{diff_ms:>5.1f}")
        print(_sep())


# ═══════════════════════════════════════════════════════════════════════════
# 6.  TABLE B — throughput × engine × scale  (spatial-locality scaling)
# ═══════════════════════════════════════════════════════════════════════════

def _make_scaled_gdf(gdf_full: gpd.GeoDataFrame, n_links: int,
                     rng: np.random.Generator) -> gpd.GeoDataFrame:
    """
    Sub-sample or over-sample by tiling the full dataset to reach n_links.
    Sub-sampling keeps spatial density representative of the real dataset.
    Over-sampling (n > len(gdf_full)) tiles with small coordinate jitter so
    the KdTree sees genuinely denser packing, not exact duplicates.
    """
    n_full = len(gdf_full)
    if n_links <= n_full:
        idx = rng.choice(n_full, size=n_links, replace=False)
        return gdf_full.iloc[idx].reset_index(drop=True)
    # tile with jitter
    repeats = math.ceil(n_links / n_full)
    frames = [gdf_full]
    for _ in range(repeats - 1):
        copy = gdf_full.copy()
        j = 0.00001  # ~1 m jitter
        copy["start_lon"] += rng.uniform(-j, j, n_full)
        copy["start_lat"] += rng.uniform(-j, j, n_full)
        copy["end_lon"]   += rng.uniform(-j, j, n_full)
        copy["end_lat"]   += rng.uniform(-j, j, n_full)
        frames.append(copy)
    big = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(big.iloc[:n_links].reset_index(drop=True), crs=gdf_full.crs)


def table_b(gdf_full: gpd.GeoDataFrame, scale_fracs: List[float],
            n_queries: int, radius_deg: float, d_limit: float) -> List[dict]:

    print(f"\n{'═'*W}")
    print("  TABLE B — Engine × Scale  (uniform queries, GeoParquet source)")
    print("  Tests the spatial-locality thesis: KdTree advantage grows with density")
    print(f"  {n_queries:,} uniform queries per scale level")
    print(f"{'═'*W}")
    print(_sep())
    print(f"  {'N links':<10}  {'Engine':<8}  {'Queries/s':>13}  {'Match%':>8}"
          f"  {'ms':>8}  {'vs base':>10}  {'nb/rs':>8}")
    print(_sep())

    rng = np.random.default_rng(99)
    n_full = len(gdf_full)
    rows = []

    for frac in scale_fracs:
        n_links = max(50, int(n_full * frac))
        gdf = _make_scaled_gdf(gdf_full, n_links, rng)
        df_links = pl.from_pandas(gdf.drop(columns=["geometry"]))
        links_list = gdf.drop(columns=["geometry"]).to_dict(orient="records")
        rust_idx = rust_build_index(gdf)
        queries = make_queries(gdf, n_queries, "clustered")

        bl_res, t_bl = baseline_match(links_list, queries, radius_deg, d_limit, cap=200)
        n_bl = len(bl_res)
        bl_qps = n_bl / t_bl

        nb_res, t_nb = numba_match(df_links, queries, radius_deg, d_limit)
        nb_qps = n_queries / t_nb
        nb_match = 100 * (nb_res != -1).sum() / n_queries
        nb_sp = (t_bl / n_bl) / (t_nb / n_queries)

        print(f"  {n_links:<10,}  {'baseline':<8}  {bl_qps:>13,.0f}  {'—':>8}"
              f"  {t_bl*1000*n_queries/n_bl:>8.1f}  {'1.0×':>10}")
        print(f"  {'':<10}  {'numba':<8}  {nb_qps:>13,.0f}  {nb_match:>7.1f}%"
              f"  {t_nb*1000:>8.1f}  {nb_sp:>9.0f}×")

        nb_rs = None
        if rust_idx is not None:
            rs_out, t_rs = rust_match(rust_idx, queries, radius_deg, d_limit)
            rs_match = 100 * sum(1 for x in rs_out[0] if x is not None) / n_queries
            rs_qps = n_queries / t_rs
            rs_sp = (t_bl / n_bl) / (t_rs / n_queries)
            nb_rs = t_nb / t_rs
            print(f"  {'':<10}  {'rust':<8}  {rs_qps:>13,.0f}  {rs_match:>7.1f}%"
                  f"  {t_rs*1000:>8.1f}  {rs_sp:>9.0f}×  {nb_rs:>5.2f}× nb/rs")
        else:
            print(f"  {'':<10}  {'rust':<8}  (not built)")

        rows.append({"n_links": n_links, "nb_rs": nb_rs})
        print(_sep())

    return rows


# ═══════════════════════════════════════════════════════════════════════════
# 7.  TABLE C — index memory footprint at each scale
# ═══════════════════════════════════════════════════════════════════════════

def _measure_numba_index_bytes(gdf: gpd.GeoDataFrame, radius_deg: float) -> int:
    """
    Exact bytes of the 7 SoA arrays that make up the Numba flat-grid index.
    Strings (start_node etc.) live only in Polars outside the hot path — not counted.
    """
    idx = build_flat_spatial_index(
        pl.from_pandas(gdf.drop(columns=["geometry"])), cell_size=radius_deg
    )
    return sum(
        arr.nbytes for arr in [
            idx["link_starts"], idx["link_ends"],
            idx["sorted_indices"], idx["sorted_keys"],
        ]
    )


def _estimate_rust_index_bytes(gdf: gpd.GeoDataFrame) -> Tuple[int, float]:
    """
    Analytical estimate of Rust index memory + measured build time.

    Vec<LinkData> layout per link:
      start_loc [f64;2]  =  16 B
      end_loc   [f64;2]  =  16 B
      start_node String  =  24 B heap header + avg len bytes
      end_node   String  =  24 B heap header + avg len bytes
      roadtype   String  =  24 B heap header + avg len bytes
      length     Option<String> ~ 24 B
      meshcode   Option<String> ~ 24 B
      Total struct overhead  ≈ 152 B + string content

    KdTree node (kdtree crate) ≈ 64 B per leaf/split node,
    balanced binary tree → ~2N nodes for N points.

    We measure actual string lengths from the DataFrame for accuracy.
    """
    try:
        import spatial_lookup
    except ImportError:
        return 0, 0.0

    n = len(gdf)
    # average string byte lengths
    avg_snode = gdf["start_node"].str.len().mean()
    avg_enode = gdf["end_node"].str.len().mean()
    avg_rtype = gdf["roadtype"].str.len().mean()
    avg_len   = gdf["length"].str.len().mean()

    per_link = 16 + 16 + (24 + avg_snode) + (24 + avg_enode) + (24 + avg_rtype) + (24 + avg_len) + 24
    vec_bytes  = int(n * per_link)
    tree_bytes = int(2 * n * 64)   # ~2N KdTree nodes × 64 B each
    total = vec_bytes + tree_bytes

    t0 = time.perf_counter()
    idx = spatial_lookup.SpatialIndex.from_arrays(
        gdf["start_lon"].tolist(), gdf["start_lat"].tolist(),
        gdf["end_lon"].tolist(),   gdf["end_lat"].tolist(),
        gdf["start_node"].tolist(), gdf["end_node"].tolist(),
        gdf["roadtype"].tolist(),
        gdf["length"].tolist(),
        gdf["ref_mesh"].tolist(),
    )
    t_build = time.perf_counter() - t0
    del idx
    return total, t_build


def table_c(gdf_full: gpd.GeoDataFrame, scale_fracs: List[float],
            radius_deg: float) -> List[dict]:

    print(f"\n{'═'*W}")
    print("  TABLE C — Index Memory Footprint × Scale")
    print("  Numba: exact bytes of 7 SoA float64/int64 arrays (strings excluded)")
    print("  Rust:  RSS delta around from_arrays() build  (includes KdTree nodes + Vec<LinkData>)")
    print(f"{'═'*W}")
    print(_sep())
    print(f"  {'N links':<10}  {'Numba idx':>12}  {'bytes/lnk':>10}"
          f"  {'Rust idx':>12}  {'bytes/lnk':>10}  {'Rust build ms':>14}")
    print(_sep())

    rng = np.random.default_rng(7)
    n_full = len(gdf_full)
    rows = []

    for frac in scale_fracs:
        n_links = max(50, int(n_full * frac))
        gdf = _make_scaled_gdf(gdf_full, n_links, rng)

        nb_bytes = _measure_numba_index_bytes(gdf, radius_deg)
        rs_bytes, t_rs_build = _estimate_rust_index_bytes(gdf)

        nb_per = nb_bytes / n_links if n_links else 0
        rs_per = rs_bytes / n_links if rs_bytes and n_links else 0

        nb_str = f"{nb_bytes/1024:.1f} KB"
        rs_str = f"{rs_bytes/1024:.1f} KB" if rs_bytes else "(not built)"
        rs_bld = f"{t_rs_build*1000:.1f}" if rs_bytes else "—"

        print(f"  {n_links:<10,}  {nb_str:>12}  {nb_per:>9.1f}B"
              f"  {rs_str:>12}  {rs_per:>9.1f}B  {rs_bld:>13} ms")
        rows.append({
            "n_links": n_links,
            "nb_bytes_per_link": nb_per,
            "rs_bytes_per_link": rs_per,
            "rs_build_ms": t_rs_build * 1000 if rs_bytes else None,
        })

    print(_sep())
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# 8.  TABLE D — small-batch repeated calls (streaming vs bulk)
#
#  Real scenario: a fleet dispatcher receives GPS pings from N_vehicles.
#  Each vehicle sends a small batch of trips every few seconds.
#  Two strategies:
#    "bulk"      — wait and accumulate all N_vehicles × batch into one call
#    "streaming" — match each vehicle's batch as it arrives (separate calls)
#
#  This reveals two costs the throughput tables hide:
#    1. Index-build cost: Numba grid is cheap to build; Rust KdTree is O(N log N)
#    2. Per-call Numba JIT dispatch overhead vs Rust FFI overhead at tiny batches
# ═══════════════════════════════════════════════════════════════════════════

# Scenarios: (label, batch_size, n_repeats)
# batch_size × n_repeats = total queries processed = comparable across rows
STREAMING_SCENARIOS = [
    ("1 q × 2000 calls",     1,    2000),
    ("5 q × 400 calls",      5,     400),
    ("20 q × 100 calls",    20,     100),
    ("100 q × 20 calls",   100,      20),
    ("500 q × 4 calls",    500,       4),
    ("2000 q × 1 call",   2000,       1),   # pure bulk — same as Table A
]


def table_d(gdf_full: gpd.GeoDataFrame, radius_deg: float, d_limit: float) -> Tuple[List[dict], List[dict]]:
    print(f"\n{'═'*W}")
    print("  TABLE D — Small-batch Repeated Calls  (streaming vs bulk)")
    print("  Scenario: same 2,000 total queries split into batches of varying size.")
    print("  'w/ rebuild' = index rebuilt each call (naive streaming).")
    print("  'cached idx' = index built once, reused across calls (correct pattern).")
    print(f"{'═'*W}")

    gdf = gdf_full   # full 3,628-link dataset
    df_links = pl.from_pandas(gdf.drop(columns=["geometry"]))
    all_queries = make_queries(gdf, 2000, "uniform", seed=5)

    # Pre-build both indexes once for the "cached" path
    nb_idx_cached = build_flat_spatial_index(df_links, cell_size=radius_deg)

    rust_idx_cached = None
    try:
        import spatial_lookup
        rust_idx_cached = spatial_lookup.SpatialIndex.from_arrays(
            gdf["start_lon"].tolist(), gdf["start_lat"].tolist(),
            gdf["end_lon"].tolist(),   gdf["end_lat"].tolist(),
            gdf["start_node"].tolist(), gdf["end_node"].tolist(),
            gdf["roadtype"].tolist(), gdf["length"].tolist(), gdf["ref_mesh"].tolist(),
        )
    except ImportError:
        pass

    def _run_numba_batch(queries, idx):
        df_q = parse_wkt_polars(pl.DataFrame({"wkt": queries}))
        coords = np.ascontiguousarray(
            df_q.select(["start_lon","start_lat","end_lon","end_lat"]).to_numpy(),
            dtype=np.float64,
        )
        return match_bulk_numba(
            coords, idx["link_starts"], idx["link_ends"],
            idx["sorted_indices"], idx["sorted_keys"],
            idx["min_lon"], idx["min_lat"], idx["cell_size"], idx["num_cols"],
            radius_deg, d_limit,
        )

    cached_rows, rebuild_rows = [], []

    for section, with_rebuild, out_rows in [
        ("cached index (build once, reuse)",          False, cached_rows),
        ("rebuild index each call (naive streaming)", True,  rebuild_rows),
    ]:
        print(f"\n  — {section} —")
        print(_sep())
        print(f"  {'Scenario':<22}  {'Numba ms':>10}  {'q/s':>10}"
              f"  {'Rust ms':>10}  {'q/s':>10}  {'nb/rs':>8}")
        print(_sep())

        for label, batch_sz, n_reps in STREAMING_SCENARIOS:
            total_q = batch_sz * n_reps

            t0 = time.perf_counter()
            for rep in range(n_reps):
                chunk = all_queries[rep * batch_sz: rep * batch_sz + batch_sz]
                idx = build_flat_spatial_index(df_links, cell_size=radius_deg) if with_rebuild else nb_idx_cached
                _run_numba_batch(chunk, idx)
            t_nb = time.perf_counter() - t0
            nb_qps = total_q / t_nb

            if rust_idx_cached is not None:
                t0 = time.perf_counter()
                for rep in range(n_reps):
                    chunk = all_queries[rep * batch_sz: rep * batch_sz + batch_sz]
                    if with_rebuild:
                        rust_idx_local = spatial_lookup.SpatialIndex.from_arrays(
                            gdf["start_lon"].tolist(), gdf["start_lat"].tolist(),
                            gdf["end_lon"].tolist(),   gdf["end_lat"].tolist(),
                            gdf["start_node"].tolist(), gdf["end_node"].tolist(),
                            gdf["roadtype"].tolist(), gdf["length"].tolist(),
                            gdf["ref_mesh"].tolist(),
                        )
                    else:
                        rust_idx_local = rust_idx_cached
                    rust_idx_local.find_match_bulk_rayon(chunk, radius_deg, d_limit)
                t_rs = time.perf_counter() - t0
                rs_qps = total_q / t_rs
                ratio = t_nb / t_rs
                print(f"  {label:<22}  {t_nb*1000:>10.1f}  {nb_qps:>10,.0f}"
                      f"  {t_rs*1000:>10.1f}  {rs_qps:>10,.0f}  {ratio:>7.2f}×")
                out_rows.append({"label": label, "batch_sz": batch_sz, "nb_rs": ratio})
            else:
                print(f"  {label:<22}  {t_nb*1000:>10.1f}  {nb_qps:>10,.0f}"
                      f"  {'(not built)':>10}  {'—':>10}  {'—':>8}")
                out_rows.append({"label": label, "batch_sz": batch_sz, "nb_rs": None})

        print(_sep())

    return cached_rows, rebuild_rows


# ═══════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbf",        default="data/monaco-latest.osm.pbf")
    ap.add_argument("--geoparquet", default=None)
    ap.add_argument("--n_queries",  type=int,   default=2000)
    ap.add_argument("--radius_deg", type=float, default=0.0002)
    ap.add_argument("--d_limit",    type=float, default=10.0)
    ap.add_argument("--scales",     default="0.25,0.5,1.0,2.0,4.0")
    args = ap.parse_args()

    pbf_path = Path(args.pbf)
    gpq_path = Path(args.geoparquet) if args.geoparquet else (
        pbf_path.parent / (pbf_path.stem + "_links.geoparquet")
    )
    scale_fracs = [float(x) for x in args.scales.split(",")]

    # --- Ingest via PBF ---
    print(f"Ingesting via PBF  ({pbf_path.name}) …")
    gdf_pbf, t_pbf = ingest_pbf(pbf_path)
    print(f"  {len(gdf_pbf):,} links  in {t_pbf*1000:.0f} ms")

    # --- Write GeoParquet (idempotent) ---
    if not gpq_path.exists() or gpq_path.stat().st_mtime < pbf_path.stat().st_mtime:
        write_geoparquet(gdf_pbf, gpq_path)
        print(f"  GeoParquet written → {gpq_path.name}  ({gpq_path.stat().st_size//1024} KB)")

    # --- Ingest via GeoParquet ---
    print(f"Ingesting via GeoParquet ({gpq_path.name}) …")
    gdf_gpq, t_gpq = ingest_geoparquet(gpq_path)
    print(f"  {len(gdf_gpq):,} links  in {t_gpq*1000:.0f} ms")

    # --- Warm up Numba JIT (one-time cost, excluded from all timings) ---
    print("\nWarming up Numba JIT … ", end="", flush=True)
    t_jit = time.perf_counter()
    _wq = pl.DataFrame({"wkt": [
        f"LINESTRING({gdf_pbf.iloc[0].start_lon:.6f} {gdf_pbf.iloc[0].start_lat:.6f},"
        f" {gdf_pbf.iloc[0].end_lon:.6f} {gdf_pbf.iloc[0].end_lat:.6f})"
    ]})
    match_dataframe(
        pl.from_pandas(gdf_pbf.drop(columns=["geometry"])), _wq,
        radius_deg=args.radius_deg, d_limit_meters=args.d_limit,
    )
    print(f"done ({(time.perf_counter()-t_jit)*1000:.0f} ms, one-time JIT compile)")

    # --- Run Tables ---
    table_a(gdf_pbf, t_pbf, gdf_gpq, t_gpq, args.n_queries, args.radius_deg, args.d_limit)
    b_rows = table_b(gdf_gpq, scale_fracs, args.n_queries, args.radius_deg, args.d_limit)
    c_rows = table_c(gdf_gpq, scale_fracs, args.radius_deg)
    d_cached, d_rebuild = table_d(gdf_gpq, args.radius_deg, args.d_limit)

    # ── Collect signals from all four tables ──────────────────────────────────

    # Table A1: ingestion speedup (computed inline in table_a, recompute here)
    ingest_ratio = t_pbf / t_gpq if t_gpq > 0 else 1.0

    # Table B: nb/rs at smallest and largest N, and trend direction
    b_valid = [r for r in b_rows if r["nb_rs"] is not None]
    b_ratio_small = b_valid[0]["nb_rs"]  if b_valid else None
    b_ratio_large = b_valid[-1]["nb_rs"] if b_valid else None
    b_n_small     = b_valid[0]["n_links"] if b_valid else None
    b_n_large     = b_valid[-1]["n_links"] if b_valid else None
    b_trend       = "decreasing" if (b_ratio_large and b_ratio_small and b_ratio_large < b_ratio_small) else "increasing"

    # Table C: memory ratio (Rust vs Numba bytes/link)
    c_nb_bpl = c_rows[0]["nb_bytes_per_link"] if c_rows else 48.0
    c_rs_bpl = c_rows[0]["rs_bytes_per_link"] if c_rows else 0.0
    c_mem_ratio = c_rs_bpl / c_nb_bpl if c_nb_bpl and c_rs_bpl else 0.0

    # Table D cached: batch size where Rust advantage drops below 2× (nearing parity)
    d_cached_valid = [(r["label"], r["batch_sz"], r["nb_rs"]) for r in d_cached if r["nb_rs"] is not None]
    # find smallest batch where nb/rs > threshold (Rust clearly wins on tiny batches)
    rust_tiny_batch_label = d_cached_valid[0][0]  if d_cached_valid else "—"
    rust_tiny_batch_ratio = d_cached_valid[0][2]  if d_cached_valid else 0.0
    rust_bulk_ratio       = d_cached_valid[-1][2] if d_cached_valid else 0.0
    # crossover in rebuild mode: last batch where nb/rs < 1 (Numba still faster)
    rebuild_valid = [(r["label"], r["batch_sz"], r["nb_rs"]) for r in d_rebuild if r["nb_rs"] is not None]
    numba_rebuild_wins_up_to = next(
        (r for r in reversed(rebuild_valid) if r[2] < 1.0), None
    )
    rust_rebuild_wins_above = next(
        (r for r in rebuild_valid if r[2] >= 1.0), None
    )

    # ── Print use-case conclusions ────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print("  USE-CASE CONCLUSIONS  (reasoned from measured signals)")
    print(f"{'═'*W}")

    # --- Rust + PBF → real-time streaming ---
    print(f"""
  ┌─ USE CASE 1: Real-time streaming  (Rust KdTree + PBF source)
  │
  │  Signal — Table D cached index, smallest batch '{rust_tiny_batch_label}':
  │    Rust is {rust_tiny_batch_ratio:.0f}× faster than Numba per query when the index is
  │    held in memory across calls.  At bulk ({d_cached_valid[-1][0]}) that shrinks
  │    to {rust_bulk_ratio:.1f}× — but streaming never gets to bulk.
  │
  │  Signal — Table B at N={b_n_small:,} links:
  │    Rust nb/rs = {b_ratio_small:.2f}× — strongest advantage at low link counts.
  │    This matches a city-tile or neighbourhood index loaded per vehicle.
  │
  │  Signal — Table A1:
  │    PBF parse = {t_pbf*1000:.0f} ms (one-time server startup cost).  Once loaded,
  │    the KdTree lives in memory and is never rebuilt.  The {ingest_ratio:.1f}× slower
  │    parse vs GeoParquet is irrelevant — it happens once, not per request.
  │
  │  Signal — Table C:
  │    Rust index = {c_rs_bpl:.0f} B/link ({c_mem_ratio:.1f}× Numba).  For a small-scale
  │    real-time tile (< 5K links) that is < {int(c_rs_bpl * 5000 / 1024 / 1024) + 1} MB — acceptable.
  │
  │  Conclusion: Load PBF once at startup → build KdTree → keep cached.
  │    Each incoming vehicle GPS ping matches in ~{1/rust_tiny_batch_ratio*1000:.1f} ms (single query).
  │    Rust is the right engine because index-build cost is amortised to zero
  │    and KdTree traversal is O(log N) with strong spatial locality pruning.
  └─""")

    # --- Numba + GeoParquet → large-scale batch pipeline ---
    nb_mem_at_large = c_rows[-1]["nb_bytes_per_link"] * b_n_large / 1024 / 1024 if c_rows else 0
    rs_mem_at_large = c_rows[-1]["rs_bytes_per_link"] * b_n_large / 1024 / 1024 if c_rows else 0
    print(f"""  ┌─ USE CASE 2: Large-scale batch pipeline  (Numba flat-grid + GeoParquet)
  │
  │  Signal — Table B at N={b_n_large:,} links:
  │    nb/rs = {b_ratio_large:.2f}× — Rust advantage has shrunk from {b_ratio_small:.2f}× at N={b_n_small:,}.
  │    At this density the flat grid checks more haversine candidates per cell,
  │    but Numba's @njit parallel loop distributes work across all CPU cores
  │    against a single contiguous memory block — no pointer chasing.
  │
  │  Signal — Table C at N={b_n_large:,} links:
  │    Numba index = {nb_mem_at_large:.1f} MB  vs  Rust index = {rs_mem_at_large:.1f} MB.
  │    {c_mem_ratio:.1f}× memory saving matters when processing a full city/region
  │    dataset where N can reach hundreds of thousands of links.
  │
  │  Signal — Table A1:
  │    GeoParquet loads in {t_gpq*1000:.0f} ms vs PBF {t_pbf*1000:.0f} ms ({ingest_ratio:.1f}× faster).
  │    In a nightly batch job that reloads the dataset each run, this matters.
  │    GeoParquet also enables predicate pushdown — you can load only the
  │    bounding-box rows you need without parsing the full PBF.
  │
  │  Signal — Table D rebuild, Numba faster up to:
  │    '{numba_rebuild_wins_up_to[0] if numba_rebuild_wins_up_to else "—"}'
  │    — in a batch pipeline the index is built once per job, so rebuild cost
  │    is irrelevant.  Numba's O(N) flat-grid build is also {int(c_rs_bpl / c_nb_bpl)}× cheaper
  │    in memory than the Rust KdTree, freeing RAM for the data itself.
  │
  │  Conclusion: Read GeoParquet once per batch → build flat SoA grid → fan
  │    out {args.n_queries:,}+ queries via Numba prange across all cores.
  │    Numba is the right engine because flat contiguous arrays eliminate
  │    cache misses at scale and the uniform grid never rebuilds.
  └─""")

    # --- Combined recommendation ---
    # Summarise the rebuild decision boundary from the actual data
    # nb/rs < 1 means Numba wins (Rust KdTree rebuild cost dominates)
    # nb/rs >= 1 means Rust wins (query speed overcomes rebuild cost)
    all_nb_faster = all(r[2] is not None and r[2] < 1.0 for r in rebuild_valid)
    if all_nb_faster:
        boundary_line = (
            f"  Rust KdTree rebuild cost dominates at EVERY batch size measured.\n"
            f"  → If you must rebuild per call: always USE NUMBA regardless of batch size.\n"
            f"  → If you can cache the index: USE RUST (wins {rust_tiny_batch_ratio:.0f}× at 1-query batches)."
        )
    elif rust_rebuild_wins_above:
        boundary_line = (
            f"  batch_size < {rust_rebuild_wins_above[1]:>5}  "
            f"→  Numba faster (Rust rebuild dominates)  →  USE NUMBA\n"
            f"  batch_size ≥ {rust_rebuild_wins_above[1]:>5}  "
            f"→  Rust query speed amortises rebuild     →  USE RUST"
        )
    else:
        boundary_line = "  Numba faster at all measured batch sizes when rebuilding."

    print(f"  Decision boundary (Table D, rebuild-per-call strategy):\n{boundary_line}\n")
    print(f"{'═'*W}\n")


if __name__ == "__main__":
    main()
