# HL-Optimizer
Tools to audit high-level programming identifying necessity for parallelization and lower level instructions.

## `hl_audit.py`

A stdlib-only Python script that audits a target Python file and classifies
each hot function as one of:

| Verdict | Meaning |
|---|---|
| `REWRITE_LOWLEVEL` | Bottleneck is CPython's `PyObject` layout — pointer chasing and cache misses. Adding cores will *parallelize the cache misses*, not fix them. Move to NumPy / Rust (PyO3) / C. |
| `CPU_PARALLELIZE` | Hot, GIL-releasing, no shared mutable state. `multiprocessing.Pool` will actually scale. |
| `CPU_PARALLELIZE_CAUTION` | Parallelizable but writes shared state — refactor to a pure function first. |
| `ASYNC_OR_THREADS` | I/O-bound. Cores don't help; use `asyncio` or a `ThreadPoolExecutor`. |
| `LEAVE_ALONE` | Not actually hot. |

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
