"""
get_ids.py  —  Distributed GPS-trip → OSM-link ID matching pipeline.

Architecture
~~~~~~~~~~~~
Polars columnar WKT parse  →  flat SoA spatial index  →  Numba @njit parallel
kernel  →  Dask distributed fan-out  →  Polars integer-index join back to
link metadata strings.

The two engines (this file vs. src/lib.rs) use deliberately different memory
layouts so they can be benchmarked head-to-head.  See engines/numba_engine.py
for the SoA data-shape documentation.

Usage
~~~~~
    python get_ids.py \\
        --links_parquet links.parquet \\
        --input_csv     trips.csv \\
        --output        matched.csv

links.parquet must have columns:
    start_lon, start_lat, end_lon, end_lat  (Float64)
    start_node, end_node, roadtype          (Utf8, optional extras kept as-is)

trips.csv must have a 'wkt' column with LINESTRING geometries.
"""

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

    print(f"Saving results to {args.output} …")
    df_final.write_csv(args.output)
    print(f"Done. Elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
