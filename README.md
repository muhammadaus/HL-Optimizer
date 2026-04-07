# HL-Optimizer
Tools to audit high-level programming identifying necessity for parallelization and lower level instructions.

## `hl_audit.py`

A stdlib-only Python script that audits a target Python file and classifies
each hot function as one of:

| Verdict | Meaning |
|---|---|
| `REWRITE_LOWLEVEL` | Bottleneck is CPython's `PyObject` layout — pointer chasing and cache misses. Adding cores will *parallelize the cache misses*, not fix them. Move to NumPy / Rust (PyO3) / C. |
| `THREADING_BROKEN` | Function dispatches to `threading.Thread` / `ThreadPoolExecutor` but the worker is pure-Python CPU work. The GIL is serializing it — your threads are taking turns. Switch to `ProcessPoolExecutor`. |
| `USE_JOIN` | Quadratic string concatenation: `s += x` inside a loop. Replace with `''.join(parts)`. |
| `PANDAS_BATCH` | DataFrame is being grown one row at a time inside a loop (`pd.concat([df, …])` / `df.append`). Build a list of row dicts and call `pd.DataFrame(rows)` once. |
| `CPU_PARALLELIZE` | Hot, GIL-releasing, no shared mutable state. `multiprocessing.Pool` will actually scale. |
| `CPU_PARALLELIZE_CAUTION` | Parallelizable but writes shared state — refactor to a pure function first. |
| `ASYNC_OR_THREADS` | I/O-bound. Cores don't help; use `asyncio` or a `ThreadPoolExecutor`. |
| `LEAVE_ALONE` | Not actually hot. |

The classifier order is deliberate: **structural bugs first** (`THREADING_BROKEN`, `USE_JOIN`, `PANDAS_BATCH`), then **I/O**, then **locality**, then **parallelism**. Structural bugs are reported even when they're not the current bottleneck — they're scaling/correctness bugs that become the next bottleneck once you fix the dominant one.

It combines a static AST pass (hot-loop shapes, `math.sin`/`cos` in Python
loops, `list.append` inside loops, I/O calls, global-state writes) with a
dynamic pass (`cProfile` + `tracemalloc`) that measures where wall time and
allocations actually land. The classifier's decision order is deliberate:
**I/O first, locality second, parallelism last** — so a cache-miss-bound
kernel is never misclassified as "just add more cores."

### Usage

```bash
python hl_audit.py path/to/target.py --args "arg1 arg2" --min-share 0.05
python hl_audit.py path/to/target.py --json report.json
```

### Sample

[`samples/haversine_demo.py`](samples/haversine_demo.py) is a realistic
geospatial workload (1M GPS points, great-circle distances) with three
implementations in one file:

```bash
# The textbook pointer-chase: list of tuples, math.sin in a Python loop.
python hl_audit.py samples/haversine_demo.py --args "--mode pure  --n 200000"
# → haversine_all_pure classified REWRITE_LOWLEVEL (cache misses, not cycles)

# Contiguous float64 buffers, one vectorized expression.
python hl_audit.py samples/haversine_demo.py --args "--mode numpy --n 200000"
# → kernel drops out of the hot list; 20-200x faster on one core

# Pure-Python kernel under multiprocessing.Pool.
python samples/haversine_demo.py --mode mp --n 200000
# → Disappointing speedup vs core count — exactly the mistake the auditor
#   exists to prevent you from making.
```

Run the three modes back-to-back and observe that `numpy` ≪ `mp` ⪅ `pure`.
That gap is the whole point: it proves that on cache-bound workloads,
rewriting the data layout beats adding cores by an order of magnitude or
more.

### Sample 2: nightly e-commerce accounting export

[`samples/accounting_export_demo.py`](samples/accounting_export_demo.py) is
a realistic three-stage data job — the kind of script that runs at 2am in
thousands of small-and-mid-size online retailers tonight, producing a file
of yesterday's transactions for the accounting system (QuickBooks /
NetSuite / Xero / SAP) to pick up at 7am. It runs fine at 5k
transactions/day. At 500k/day the cron starts overlapping itself, finance
gets yesterday's numbers at lunch instead of 9am, and a data engineer gets
paged.

The script as written contains three independent bugs, one per stage —
each is the *idiomatic wrong way* to write that stage under deadline
pressure:

```bash
ACCT_N=800 python hl_audit.py samples/accounting_export_demo.py
```

A single audit produces three distinct verdicts on three distinct
functions in one report:

| Stage | Function | Verdict | Fix |
|---|---|---|---|
| 1. Load CSV drops | `load_daily_drops` | `THREADING_BROKEN` | `ProcessPoolExecutor`, not `ThreadPoolExecutor` — `csv.reader` is pure Python |
| 2. Enrich rows with features | `enrich_transactions` | `PANDAS_BATCH` | Build a list of dicts, call `pd.DataFrame(rows)` once |
| 3. Emit accounting file | `emit_accounting_file` | `USE_JOIN` | Build a list of lines, call `'\n'.join(lines)` once |

This is the auditor's actual value proposition on production code: real
scripts often need **three different kinds of fix**, not just one big
rewrite. Fixing one moves the bottleneck to the next; the audit shows all
three at once so you can plan the order of attack.
