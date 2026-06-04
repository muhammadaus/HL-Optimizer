# engines package — importable spatial-matching kernels and route enrichment
from .numba_engine import (
    haversine_nb,
    match_bulk_numba,
    build_flat_spatial_index,
    parse_wkt_polars,
    match_dataframe,
)
from .route_enricher import RouteResult, enrich_single, enrich_batch

__all__ = [
    "haversine_nb",
    "match_bulk_numba",
    "build_flat_spatial_index",
    "parse_wkt_polars",
    "match_dataframe",
    "RouteResult",
    "enrich_single",
    "enrich_batch",
]
