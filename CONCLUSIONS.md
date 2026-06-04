# Benchmark Conclusions

Measured on Monaco OSM (3,628 road links, 2,000 queries).
Run: `uv run python scripts/prepare_and_benchmark.py --pbf data/monaco-latest.osm.pbf`

---

## Use Case 1 — Real-time streaming: Rust KdTree + PBF

**Pattern:** Long-running matching server. PBF loaded once at startup, KdTree
kept in memory, single GPS pings matched as they arrive.

| Signal | Value | Source |
|---|---|---|
| Rust vs Numba, 1-query batch (cached index) | **145×** faster | Table D |
| Rust vs Numba at N=907 links (city tile) | **2.2×** faster | Table B |
| PBF parse time (one-time startup) | 264 ms | Table A1 |
| Rust index size at N=5K links | ~2 MB (307 B/link) | Table C |

**Why Rust wins here:**
The KdTree is built once and never rebuilt. Each incoming vehicle ping is a
single-query call, where Rust's O(log N) adaptive tree traversal is 145×
faster than Numba's flat-grid dispatch overhead. Spatial locality pruning is
strongest at low link counts — the KdTree descends to a tight subtree in a
dense urban tile far faster than the uniform grid's fixed 3×3 cell scan.

The 1.8× slower PBF parse vs GeoParquet is irrelevant here: it happens once
at server startup and is immediately amortised across millions of requests.

**When to use:**
- Vehicle tracking / fleet dispatch (1–50 queries per GPS polling cycle)
- Real-time map matching in a persistent process
- City-tile or neighbourhood index (< 10K links per shard)

---

## Use Case 2 — Large-scale batch pipeline: Numba flat-grid + GeoParquet

**Pattern:** Nightly or hourly batch job. Full dataset reloaded from GeoParquet
each run, all trip queries matched in one parallel sweep.

| Signal | Value | Source |
|---|---|---|
| Rust advantage at N=14,512 links | **1.4×** (down from 2.2× at N=907) | Table B |
| Numba vs Rust memory per link | **48 B vs 307 B** (6.4× smaller) | Table C |
| GeoParquet vs PBF load time | **1.8×** faster (146 ms vs 264 ms) | Table A1 |
| Numba throughput at bulk (2,000 queries) | 350K+ queries/s | Table D |

**Why Numba wins here:**
As the dataset grows, the KdTree's adaptive pruning advantage compresses.
The flat uniform grid checks a fixed 3×3 cell neighbourhood, but Numba's
`@njit(parallel=True)` distributes those checks across all CPU cores against
a single contiguous float64 block — no pointer chasing, no String heap
allocations. At 14K links the gap narrows to 1.4× and Numba uses 6.4× less
RAM, freeing memory for the data itself.

GeoParquet's 1.8× faster load matters in a batch job that re-reads the
dataset on every run. It also supports predicate pushdown — loading only the
bounding-box rows needed for a regional job without parsing the full PBF.

**When to use:**
- Nightly trip matching over a city-wide or regional road network
- Historical GPS trace processing (100K+ queries per run)
- Memory-constrained environments (Numba index: 48 B/link flat)
- Pipelines already using Polars/Dask for upstream data transformations

---

## Decision Boundary

The key variable is whether the spatial index is **cached** or **rebuilt per call**.

```
Index cached across calls (long-running process):
    → USE RUST at all batch sizes
       Rust is 145× faster at 1-query batches, 1.5× at 2,000-query bulk.

Index rebuilt each call (stateless / serverless):
    → USE NUMBA at all batch sizes
       Rust KdTree rebuild is O(N log N) and dominates at every measured batch.
       Numba flat-grid rebuild is O(N) — 3× faster to rebuild, 6.4× less RAM.
```

### Query throughput (PBF vs GeoParquet — no difference)

Table A2 confirms that once data is loaded, query times from PBF and GeoParquet
are within run-to-run noise (±2 ms). The ingestion path leaves no artifact on
matching speed. **Choose your data source based on load frequency, not query
performance.**

---

## Summary Table

| Dimension | Rust KdTree | Numba flat-grid |
|---|---|---|
| Index structure | Adaptive KdTree (pointer tree) | Uniform grid (flat sorted arrays) |
| Memory per link | 307 B (AoS: Vec\<LinkData\> + String heap) | 48 B (SoA: 4 float64 arrays) |
| Index build cost | O(N log N) — expensive to rebuild | O(N) — cheap to rebuild |
| Single-query latency (cached) | **~0.005 ms** | ~0.7 ms |
| Bulk throughput (2K queries) | ~500K q/s | ~350K q/s |
| Advantage at low N (< 2K links) | **2–3×** | — |
| Advantage at high N (> 10K links) | ~1.2–1.4× | catches up |
| Best data source | PBF (parse once, serve forever) | GeoParquet (fast reload per batch) |
| Best use case | Real-time streaming, city-tile server | Batch pipeline, regional analysis |
