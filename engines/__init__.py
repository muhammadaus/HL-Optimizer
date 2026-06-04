# engines package — importable spatial-matching kernels
from .numba_engine import (
    haversine_nb,
    match_bulk_numba,
    build_flat_spatial_index,
    parse_wkt_polars,
    match_dataframe,
)

__all__ = [
    "haversine_nb",
    "match_bulk_numba",
    "build_flat_spatial_index",
    "parse_wkt_polars",
    "match_dataframe",
]
