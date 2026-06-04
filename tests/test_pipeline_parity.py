"""
tests/test_pipeline_parity.py
-----------------------------
Head-to-head benchmark: Numba/Polars/Dask engine vs. Rust/PyO3 engine.

Both engines receive *exactly* the same synthetic road-link table and WKT
query strings.  The test asserts that their matched link IDs agree on all
queries that have a known ground-truth match, then prints wall-time for each.

Rust path
~~~~~~~~~
Requires the spatial_lookup extension to be compiled first:

    maturin develop          # builds a debug build
    maturin develop --release  # builds an optimised release build

If the extension is not installed, the Rust portion is skipped gracefully;
the Numba portion still runs and must pass.

Run
~~~
    pytest tests/test_pipeline_parity.py -v -s
"""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np
import polars as pl
import pytest

from engines.numba_engine import match_dataframe


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

# Tolerance used to place "planted" queries exactly on a known link.
# Must be smaller than d_limit in the engine call below.
_PLANT_OFFSET_DEG = 0.000005   # ~0.5 m at mid-latitudes
_RADIUS_DEG       = 0.0002
_D_LIMIT_M        = 10.0

# Japan DRM bounding box — realistic coordinate range for the engine
_BASE_LON, _BASE_LAT = 139.65, 35.67


def _make_synthetic_dataset(
    n_links: int = 500,
    n_planted: int = 60,
    n_noise: int = 40,
    seed: int = 42,
) -> Tuple[pl.DataFrame, pl.DataFrame, List[int]]:
    """
    Build a deterministic synthetic links + queries dataset.

    Returns
    -------
    df_links      pl.DataFrame  — road-link table (N rows)
    df_query      pl.DataFrame  — WKT query table (n_planted + n_noise rows)
    expected_ids  List[int]     — df_links row index expected for each
                                  *planted* query (-1 means no match)
    """
    rng = np.random.default_rng(seed)

    # --- road links: random short segments in a ~1°×1° bounding box ---
    start_lons = _BASE_LON + rng.uniform(0, 0.5, n_links)
    start_lats = _BASE_LAT + rng.uniform(0, 0.5, n_links)
    # Each link extends ~5–50 m in a random compass direction
    seg_len_deg = rng.uniform(0.00005, 0.0005, n_links)
    bearing     = rng.uniform(0, 2 * np.pi, n_links)
    end_lons = start_lons + seg_len_deg * np.cos(bearing)
    end_lats = start_lats + seg_len_deg * np.sin(bearing)

    df_links = pl.DataFrame({
        "start_lon":  start_lons,
        "start_lat":  start_lats,
        "end_lon":    end_lons,
        "end_lat":    end_lats,
        "start_node": [f"S{i:04d}" for i in range(n_links)],
        "end_node":   [f"E{i:04d}" for i in range(n_links)],
        "roadtype":   ["residential"] * n_links,
        "length":     [f"{seg_len_deg[i]*111000:.1f}" for i in range(n_links)],
        "meshcode":   ["5339" for _ in range(n_links)],
    })

    # --- planted queries: start/end slightly offset from known link endpoints ---
    # We pick n_planted distinct links and nudge both endpoints by < _PLANT_OFFSET_DEG
    chosen = rng.choice(n_links, size=n_planted, replace=False)
    noise_s = rng.uniform(-_PLANT_OFFSET_DEG, _PLANT_OFFSET_DEG, (n_planted, 2))
    noise_e = rng.uniform(-_PLANT_OFFSET_DEG, _PLANT_OFFSET_DEG, (n_planted, 2))

    q_slons = start_lons[chosen] + noise_s[:, 0]
    q_slats = start_lats[chosen] + noise_s[:, 1]
    q_elons = end_lons[chosen]   + noise_e[:, 0]
    q_elats = end_lats[chosen]   + noise_e[:, 1]

    planted_wkt = [
        f"LINESTRING({q_slons[i]:.6f} {q_slats[i]:.6f}, {q_elons[i]:.6f} {q_elats[i]:.6f})"
        for i in range(n_planted)
    ]
    expected_ids: List[int] = [int(chosen[i]) for i in range(n_planted)]

    # --- noise queries: far-away points that should produce no match ---
    noise_lons = _BASE_LON + 2.0 + rng.uniform(0, 0.1, n_noise)  # 2° away
    noise_lats = _BASE_LAT + 2.0 + rng.uniform(0, 0.1, n_noise)
    noise_wkt = [
        f"LINESTRING({noise_lons[i]:.6f} {noise_lats[i]:.6f}, "
        f"{noise_lons[i]+0.001:.6f} {noise_lats[i]+0.001:.6f})"
        for i in range(n_noise)
    ]

    all_wkt = planted_wkt + noise_wkt
    df_query = pl.DataFrame({"wkt": all_wkt})

    return df_links, df_query, expected_ids


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset():
    return _make_synthetic_dataset()


# ---------------------------------------------------------------------------
# Numba engine test
# ---------------------------------------------------------------------------

class TestNumbaEngine:
    def test_planted_matches(self, dataset):
        """Planted queries must resolve to the expected link index."""
        df_links, df_query, expected_ids = dataset

        t0 = time.perf_counter()
        match_idx = match_dataframe(
            df_links, df_query,
            radius_deg=_RADIUS_DEG,
            d_limit_meters=_D_LIMIT_M,
        )
        elapsed = time.perf_counter() - t0

        n_planted = len(expected_ids)
        matched = match_idx[:n_planted].to_list()

        print(f"\n[Numba] {n_planted} planted queries in {elapsed*1000:.1f} ms")
        for i, (got, want) in enumerate(zip(matched, expected_ids)):
            assert got == want, (
                f"Query {i}: Numba returned link {got}, expected {want}"
            )

    def test_noise_no_match(self, dataset):
        """Far-away noise queries must return -1 (no match)."""
        df_links, df_query, expected_ids = dataset
        n_planted = len(expected_ids)
        match_idx = match_dataframe(
            df_links, df_query,
            radius_deg=_RADIUS_DEG,
            d_limit_meters=_D_LIMIT_M,
        )
        noise_matches = match_idx[n_planted:].to_list()
        assert all(m == -1 for m in noise_matches), (
            f"Expected all noise queries to return -1; got {noise_matches}"
        )


# ---------------------------------------------------------------------------
# Rust engine test — skipped gracefully if extension not compiled
# ---------------------------------------------------------------------------

spatial_lookup = pytest.importorskip(
    "spatial_lookup",
    reason="spatial_lookup extension not built. Run: maturin develop --release",
)


class TestRustEngine:
    def _build_rust_index(self, df_links: pl.DataFrame):
        """Construct a Rust SpatialIndex from the Polars links DataFrame."""
        idx = spatial_lookup.SpatialIndex.from_arrays(
            df_links["start_lon"].to_list(),
            df_links["start_lat"].to_list(),
            df_links["end_lon"].to_list(),
            df_links["end_lat"].to_list(),
            df_links["start_node"].to_list(),
            df_links["end_node"].to_list(),
            df_links["roadtype"].to_list(),
            df_links["length"].to_list(),
            df_links["meshcode"].to_list(),
        )
        return idx

    def test_planted_matches(self, dataset):
        """Planted queries must resolve to the expected start_node via Rust engine."""
        df_links, df_query, expected_ids = dataset

        idx = self._build_rust_index(df_links)
        wkt_list = df_query["wkt"].to_list()

        t0 = time.perf_counter()
        (start_nodes, end_nodes, roadtypes, lengths, meshcodes) = (
            idx.find_match_bulk_rayon(wkt_list, _RADIUS_DEG, _D_LIMIT_M)
        )
        elapsed = time.perf_counter() - t0

        n_planted = len(expected_ids)
        print(f"\n[Rust]  {n_planted} planted queries in {elapsed*1000:.1f} ms")

        # Rust returns start_node strings; map expected row indices → start_node
        expected_nodes = [df_links["start_node"][i] for i in expected_ids]
        for i, (got, want) in enumerate(zip(start_nodes[:n_planted], expected_nodes)):
            assert got == want, (
                f"Query {i}: Rust returned start_node {got!r}, expected {want!r}"
            )

    def test_noise_no_match(self, dataset):
        """Far-away noise queries must return None from the Rust engine."""
        df_links, df_query, expected_ids = dataset
        n_planted = len(expected_ids)

        idx = self._build_rust_index(df_links)
        wkt_list = df_query["wkt"].to_list()
        (start_nodes, *_) = idx.find_match_bulk_rayon(wkt_list, _RADIUS_DEG, _D_LIMIT_M)

        noise_nodes = start_nodes[n_planted:]
        assert all(n is None for n in noise_nodes), (
            f"Expected all noise queries to return None; got {noise_nodes}"
        )


# ---------------------------------------------------------------------------
# Cross-engine parity test (only runs when Rust extension is available)
# ---------------------------------------------------------------------------

class TestCrossEngineParity:
    def test_same_match_indices(self, dataset):
        """
        Both engines must agree on which link index each planted query matches.

        Numba returns integer row indices; Rust returns start_node strings.
        We translate Numba indices → start_node for the comparison.
        """
        df_links, df_query, expected_ids = dataset
        n_planted = len(expected_ids)

        # Numba
        t_nb0 = time.perf_counter()
        match_idx = match_dataframe(
            df_links, df_query,
            radius_deg=_RADIUS_DEG,
            d_limit_meters=_D_LIMIT_M,
        )
        t_nb = time.perf_counter() - t_nb0

        # Rust
        idx_rust = spatial_lookup.SpatialIndex.from_arrays(
            df_links["start_lon"].to_list(), df_links["start_lat"].to_list(),
            df_links["end_lon"].to_list(),   df_links["end_lat"].to_list(),
            df_links["start_node"].to_list(), df_links["end_node"].to_list(),
            df_links["roadtype"].to_list(),
            df_links["length"].to_list(),
            df_links["meshcode"].to_list(),
        )
        t_rs0 = time.perf_counter()
        (rust_start_nodes, *_) = idx_rust.find_match_bulk_rayon(
            df_query["wkt"].to_list(), _RADIUS_DEG, _D_LIMIT_M
        )
        t_rs = time.perf_counter() - t_rs0

        print(
            f"\n[Parity] Numba: {t_nb*1000:.1f} ms | Rust: {t_rs*1000:.1f} ms"
            f" | speedup: {t_nb/t_rs:.2f}x (Rust faster when >1)"
        )

        # Translate Numba integer indices → start_node strings
        start_nodes_col = df_links["start_node"].to_list()
        numba_nodes = [
            start_nodes_col[int(i)] if i != -1 else None
            for i in match_idx[:n_planted].to_list()
        ]
        rust_nodes = rust_start_nodes[:n_planted]

        mismatches = [
            (i, numba_nodes[i], rust_nodes[i])
            for i in range(n_planted)
            if numba_nodes[i] != rust_nodes[i]
        ]
        assert not mismatches, (
            f"Engine parity failure on {len(mismatches)}/{n_planted} queries:\n"
            + "\n".join(f"  query {i}: Numba={nb!r}  Rust={rs!r}"
                        for i, nb, rs in mismatches[:10])
        )
