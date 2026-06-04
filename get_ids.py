"""
get_ids.py  —  Distributed GPS-trip → OSM-link ID matching pipeline.

Architecture
~~~~~~~~~~~~
Polars columnar WKT parse  →  flat SoA spatial index  →  Numba @njit parallel
kernel  →  Dask distributed fan-out  →  Polars integer-index join back to
link metadata strings  →  optional Mapbox route enrichment.

The two engines (this file vs. src/lib.rs) use deliberately different memory
layouts so they can be benchmarked head-to-head.  See engines/numba_engine.py
for the SoA data-shape documentation.

Usage
~~~~~
    python get_ids.py \\
        --links_parquet links.parquet \\
        --input_csv     trips.csv \\
        --output        matched.csv

    # With route enrichment (requires MAPBOX_ACCESS_TOKEN env var):
    python get_ids.py \\
        --links_parquet links.parquet \\
        --input_csv     trips.csv \\
        --output        matched.csv \\
        --enrich

links.parquet must have columns:
    start_lon, start_lat, end_lon, end_lat  (Float64)
    start_node, end_node, roadtype          (Utf8, optional extras kept as-is)

trips.csv must have a 'wkt' column with LINESTRING geometries.

Enriched output adds per-row columns:
    distance_meters, duration_seconds, traffic_duration_seconds,
    congestion (JSON list), route_geometry (encoded polyline),
    tolls, enrichment_error
"""

import json
import os
import argparse
import time

import numpy as np
import pandas as pd
import polars as pl
import dask.dataframe as dd
from dask.distributed import Client

from engines.numba_engine import (
    build_flat_spatial_index,
    match_bulk_numba,
    parse_wkt_polars,
)
from engines.route_enricher import enrich_batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distributed spatial lookup: Polars + Numba + Dask."
    )
    parser.add_argument("--config", default=None,
                        help="Optional YAML config (reserved for future use).")
    parser.add_argument("--input_csv", required=True,
                        help="CSV with a 'wkt' column of LINESTRING geometries.")
    parser.add_argument("--links_parquet", required=True,
                        help="Parquet link table (start_lon/lat, end_lon/lat, …).")
    parser.add_argument("--output", required=True,
                        help="Output CSV path for matched results.")
    parser.add_argument("--enrich", action="store_true",
                        help="Enrich matched pairs with Mapbox routing metrics "
                             "(requires MAPBOX_ACCESS_TOKEN env var).")
    parser.add_argument("--enrich_mode", default="driving-traffic",
                        help="Mapbox routing profile (default: driving-traffic).")
    args = parser.parse_args()

    approx_radius: float = 0.0002   # degrees — candidate pre-filter radius
    d_limit: float = 10.0           # metres  — max haversine distance per endpoint

    # ------------------------------------------------------------------
    # Start Dask cluster (one process per core, no GIL contention because
    # match_bulk_numba releases it via @njit)
    # ------------------------------------------------------------------
    client = Client(n_workers=os.cpu_count(), threads_per_worker=1)
    print(f"Dask dashboard: {client.dashboard_link}")

    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load link index via Polars and build the SoA flat grid
    # ------------------------------------------------------------------
    print("Loading links …")
    df_links = pl.read_parquet(args.links_parquet)
    idx_meta = build_flat_spatial_index(df_links, cell_size=approx_radius)

    # Broadcast index arrays to all Dask workers (avoids per-task serialisation)
    idx_bcast = client.scatter(idx_meta, broadcast=True)

    # ------------------------------------------------------------------
    # 2. Load and parse query CSV with Polars (vectorised regex, no loops)
    # ------------------------------------------------------------------
    print("Parsing query WKT …")
    df_query = pl.read_csv(args.input_csv)
    df_query = parse_wkt_polars(df_query)

    # ------------------------------------------------------------------
    # 3. Distribute query partitions across the cluster via Dask
    # ------------------------------------------------------------------
    ddf_query = dd.from_pandas(
        df_query.to_pandas(),
        npartitions=os.cpu_count() * 2,
    )

    def process_partition(pdf_part: pd.DataFrame, index: dict) -> pd.Series:
        """Per-partition Numba dispatch.  Runs inside each Dask worker."""
        coords = np.ascontiguousarray(
            pdf_part[["start_lon", "start_lat", "end_lon", "end_lat"]].to_numpy(),
            dtype=np.float64,
        )
        matches = match_bulk_numba(
            coords,
            index["link_starts"], index["link_ends"],
            index["sorted_indices"], index["sorted_keys"],
            index["min_lon"], index["min_lat"],
            index["cell_size"], index["num_cols"],
            approx_radius, d_limit,
        )
        return pd.Series(matches, index=pdf_part.index)

    print("Executing distributed Numba query pipeline …")
    res_indices: pd.Series = ddf_query.map_partitions(
        process_partition,
        index=idx_bcast,
        meta=("match_idx", "int64"),
    ).compute()

    # ------------------------------------------------------------------
    # 4. Polars integer-index join: map match_idx → link metadata columns
    # ------------------------------------------------------------------
    df_query = df_query.with_columns(pl.Series("match_idx", res_indices))

    df_links_indexed = df_links.with_row_index("row_idx")
    df_final = df_query.join(
        df_links_indexed,
        left_on="match_idx",
        right_on="row_idx",
        how="left",
    )

    # ------------------------------------------------------------------
    # 5. Optional: enrich matched pairs with Mapbox routing metrics
    #    Only rows where a link was matched (match_idx >= 0) are enriched.
    #    Results are cached by geohash key — repeated endpoint pairs are free.
    # ------------------------------------------------------------------
    if args.enrich:
        print("Enriching matched pairs via Mapbox Directions API …")

        # Build the list of (origin, destination) from the matched coordinates.
        # start_lon/lat are the query trip endpoints already parsed in step 2;
        # they are the two points we want to route between.
        matched_mask = df_final["match_idx"].cast(pl.Int64) >= 0

        origins      = df_final.filter(matched_mask).select(["start_lon", "start_lat"])
        destinations = df_final.filter(matched_mask).select(["end_lon",   "end_lat"])

        pairs = [
            {
                "origin":      [origins[i, "start_lon"], origins[i, "start_lat"]],
                "destination": [destinations[i, "end_lon"], destinations[i, "end_lat"]],
            }
            for i in range(len(origins))
        ]

        route_results = enrich_batch(pairs, mode=args.enrich_mode)

        # Expand results into flat columns aligned to df_final row positions
        distance_m, duration_s, traffic_s, congestion_j, geometry, tolls_b, errors = (
            [], [], [], [], [], [], []
        )
        result_iter = iter(route_results)
        for matched in df_final["match_idx"].cast(pl.Int64).to_list():
            if matched >= 0:
                r = next(result_iter)
                distance_m.append(r.distance_meters)
                duration_s.append(r.duration_seconds)
                traffic_s.append(r.traffic_duration_seconds)
                congestion_j.append(json.dumps(r.congestion) if r.congestion else None)
                geometry.append(r.route_geometry)
                tolls_b.append(r.tolls)
                errors.append(r.error)
            else:
                distance_m.append(None); duration_s.append(None)
                traffic_s.append(None);  congestion_j.append(None)
                geometry.append(None);   tolls_b.append(None)
                errors.append(None)

        df_final = df_final.with_columns([
            pl.Series("distance_meters",          distance_m),
            pl.Series("duration_seconds",         duration_s),
            pl.Series("traffic_duration_seconds", traffic_s),
            pl.Series("congestion",               congestion_j),
            pl.Series("route_geometry",           geometry),
            pl.Series("tolls",                    tolls_b),
            pl.Series("enrichment_error",         errors),
        ])
        n_enriched = sum(1 for e in errors if e is None and distance_m[errors.index(e)] is not None)
        print(f"  Enriched {n_enriched:,} / {matched_mask.sum():,} matched rows.")

    print(f"Saving results to {args.output} …")
    df_final.write_csv(args.output)
    print(f"Done. Elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
