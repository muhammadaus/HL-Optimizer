# HL-Optimizer

Two things in one repo:

1. **`hl_audit.py`** — a stdlib-only Python auditor that classifies hot functions and tells you *which* fix applies: threading bug, string concat bug, pandas anti-pattern, cache-miss rewrite, or real parallelism.
2. **Two spatial-matching engines** built as the concrete answer to `REWRITE_LOWLEVEL` — GPS-trip endpoints → OSM road-link IDs, benchmarked head-to-head with real OSM data, plus a Mapbox route enricher that turns matched endpoints into full routing metrics.

---

## Repository layout

```
hl_audit.py                      # stdlib-only auditor (no dependencies)
get_ids.py                       # Numba/Polars/Dask batch pipeline (entry point)
engines/
  numba_engine.py                # SoA flat-grid @njit kernel + Polars WKT parse
  route_enricher.py              # Mapbox route enrichment + geohash cache
src/
  lib.rs                         # Rust/PyO3 KdTree engine (spatial_lookup extension)
scripts/
  prepare_and_benchmark.py       # Full benchmark: PBF vs GeoParquet, scale, memory, streaming
tests/
  test_pipeline_parity.py        # Head-to-head correctness + timing (pytest)
samples/
  haversine_demo.py              # REWRITE_LOWLEVEL demo: pure vs mp vs numpy
  accounting_export_demo.py      # Three-bug demo: THREADING_BROKEN + PANDAS_BATCH + USE_JOIN
data/
  monaco-latest.osm.pbf          # Real OSM extract (Monaco, ~665 KB) — download below
CONCLUSIONS.md                   # Use-case recommendations derived from benchmark numbers
Cargo.toml                       # Rust build manifest (maturin/PyO3)
pyproject.toml                   # Python build backend (maturin)
requirements.txt                 # Runtime Python deps
requirements-dev.txt             # + pytest, maturin
```

---

## Install

```bash
# Python pipeline — use uv (manages the venv that includes spatial_lookup)
uv sync

# Rust extension — requires the Rust toolchain (https://rustup.rs)
maturin develop --release        # builds spatial_lookup.so into the uv venv

# Download Monaco OSM data for the benchmark
mkdir -p data
curl -L https://download.geofabrik.de/europe/monaco-latest.osm.pbf \
     -o data/monaco-latest.osm.pbf
```

All Python commands should be run with `uv run python …` so they pick up the
venv where `spatial_lookup` is installed.

---

## 1 — `hl_audit.py` (auditor)

A stdlib-only script that classifies each hot function in a target `.py` file.
No install step — runs anywhere Python 3.8+ is available.

| Verdict | Meaning |
|---|---|
| `REWRITE_LOWLEVEL` | Bottleneck is CPython's `PyObject` layout — pointer chasing and cache misses. Adding cores parallelizes the cache misses, not the work. Move to NumPy / Numba / Rust. |
| `THREADING_BROKEN` | `ThreadPoolExecutor` / `threading.Thread` on a pure-Python CPU worker. The GIL serializes it — your threads take turns. Switch to `ProcessPoolExecutor`. |
| `USE_JOIN` | Quadratic string concat: `s += x` inside a loop. Replace with `''.join(parts)`. |
| `PANDAS_BATCH` | DataFrame grown one row at a time in a loop. Build a list of dicts, call `pd.DataFrame(rows)` once. |
| `CPU_PARALLELIZE` | Hot, GIL-releasing, no shared state. `multiprocessing.Pool` will actually scale. |
| `CPU_PARALLELIZE_CAUTION` | Parallelizable but writes shared state — refactor to a pure function first. |
| `ASYNC_OR_THREADS` | I/O-bound. Cores don't help; use `asyncio` or a thread pool. |
| `LEAVE_ALONE` | Not actually hot. |

Classifier order: **structural bugs first**, then I/O, then locality, then
parallelism — so a cache-miss-bound kernel is never misclassified as
"just add more cores."

```bash
python hl_audit.py path/to/target.py --args "arg1 arg2" --min-share 0.05
python hl_audit.py path/to/target.py --json report.json
```

### Demos

```bash
# REWRITE_LOWLEVEL: textbook pointer-chase (list of tuples, math.sin in a loop)
python hl_audit.py samples/haversine_demo.py --args "--mode pure --n 200000"

# After fix: contiguous float64, one vectorized expression — drops out of hot list
python hl_audit.py samples/haversine_demo.py --args "--mode numpy --n 200000"

# Three bugs in one file: THREADING_BROKEN + PANDAS_BATCH + USE_JOIN
ACCT_N=800 python hl_audit.py samples/accounting_export_demo.py
```

---

## 2 — Spatial-matching engines

`REWRITE_LOWLEVEL` points you at Numba or Rust. These two engines implement the
**same task** — match GPS trip endpoints to OSM road links via nearest-neighbour
search + haversine scoring — with deliberately different memory layouts so you can
benchmark the trade-off directly.

| Engine | Files | Index structure | Memory layout |
|---|---|---|---|
| **Numba / Polars / Dask** | `get_ids.py`, `engines/numba_engine.py` | Flat uniform grid + `np.searchsorted` | SoA: 7 flat `float64`/`int64` arrays, 48 B/link |
| **Rust / PyO3** | `src/lib.rs` (`spatial_lookup` extension) | Adaptive `KdTree<f64, usize, [f64;2]>` + rayon | AoS: `Vec<LinkData>` with `String` heap, 307 B/link |

Both engines read the same OSM tags: `highway`, `length`, `ref:mesh`. Node IDs
come from the way's first/last node refs — no vendor-specific tags.

### Use-case decision (from benchmark numbers — see `CONCLUSIONS.md`)

| Situation | Use |
|---|---|
| Real-time streaming, index cached across calls (server process) | **Rust** — 145× faster at 1-query batches |
| Large-scale batch pipeline, index rebuilt per job | **Numba** — 6.4× less RAM, O(N) rebuild vs O(N log N) |
| Index rebuilt per call at any batch size | **Numba** — Rust KdTree rebuild cost dominates at every measured batch size |

### Numba pipeline (`get_ids.py`)

```bash
# Match trips to links and write matched.csv
uv run python get_ids.py \
    --links_parquet data/monaco-latest.osm_links.geoparquet \
    --input_csv     trips.csv \
    --output        matched.csv

# With Mapbox route enrichment (adds distance, duration, traffic, congestion)
MAPBOX_ACCESS_TOKEN=pk.xxx uv run python get_ids.py \
    --links_parquet data/monaco-latest.osm_links.geoparquet \
    --input_csv     trips.csv \
    --output        matched.csv \
    --enrich
```

`links_parquet` columns: `start_lon`, `start_lat`, `end_lon`, `end_lat`,
`start_node`, `end_node`, `roadtype` (extras kept as-is).
`input_csv` must have a `wkt` column with `LINESTRING` geometries.

### Parity test (Numba vs Rust correctness)

```bash
# Numba only (no Rust toolchain needed)
uv run pytest tests/test_pipeline_parity.py -v -s

# Full head-to-head (after maturin develop --release)
uv run pytest tests/test_pipeline_parity.py -v -s
```

Synthetic dataset: 500 links, 60 planted queries with known matches, 40
far-away noise queries. Asserts both engines return identical link IDs and
prints wall-time for each.

---

## 3 — Route enricher (`engines/route_enricher.py`)

After spatial matching resolves two GPS endpoints to an OSM link, the enricher
calls the Mapbox Directions API and returns full routing metrics — skipping the
need to collect this data manually.

```python
from engines.route_enricher import enrich_batch

results = enrich_batch([
    {"origin": [139.7001, 35.6895], "destination": [139.7600, 35.6812]},
])
r = results[0]
# r.distance_meters, r.duration_seconds, r.traffic_duration_seconds,
# r.congestion (per-segment list), r.route_geometry, r.computed_at
```

Enriched fields per matched pair:

| Field | Type | Notes |
|---|---|---|
| `distance_meters` | int | Great-circle route distance |
| `duration_seconds` | int | Mapbox-estimated travel time |
| `traffic_duration_seconds` | int | Traffic-aware travel time (`driving-traffic` profile) |
| `congestion` | list[str] | Per-segment: `"low"`, `"moderate"`, `"heavy"`, `"severe"` |
| `route_geometry` | str | Encoded polyline (Mapbox format) |
| `tolls` | bool \| None | Not available on Mapbox free tier |
| `computed_at` | str | ISO-8601 UTC timestamp |
| `error` | str \| None | Set on API failure — pipeline always writes output |

**Cache:** keyed by `origin_geohash7 + dest_geohash7 + mode + provider + hour_bucket`
(stored in `.route_cache/`). Identical endpoint pairs within ~150 m share one API
call across runs. Set `ROUTE_CACHE_DIR` to change the cache location.

**Environment variables:**

```bash
MAPBOX_ACCESS_TOKEN=pk.xxx   # required
ROUTE_CACHE_DIR=.route_cache # optional, default: .route_cache/
```

---

## 4 — Benchmark (`scripts/prepare_and_benchmark.py`)

Runs on the real Monaco OSM extract (3,628 road links). Produces four tables:

| Table | What it shows |
|---|---|
| A1 | PBF vs GeoParquet ingestion time side-by-side (parse ms, index build ms, query ms, total ms) |
| A2 | Query throughput PBF vs GeoParquet per engine × distribution — proves ingestion path leaves no artifact on query speed |
| B | Throughput × scale (N=907→14,512 links) — nb/rs ratio shows KdTree advantage trend |
| C | Index memory footprint per link at each scale (Numba: 48 B flat; Rust: 307 B AoS+heap) |
| D | Small-batch repeated calls: cached index vs rebuild-per-call across 6 batch sizes |

Conclusions block at the end is derived from measured numbers — no hardcoded strings.

```bash
uv run python scripts/prepare_and_benchmark.py --pbf data/monaco-latest.osm.pbf

# Options
--n_queries  2000          # queries per distribution (default: 2000)
--radius_deg 0.0002        # candidate search radius in degrees
--d_limit    10.0          # max haversine distance per endpoint (metres)
--scales     0.25,0.5,1,2,4  # link-count scale factors for Table B
```

See `CONCLUSIONS.md` for the full written-up findings.
