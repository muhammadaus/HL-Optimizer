"""
Haversine distance over a synthetic GPS track.

Three implementations of the same computation, selectable via --mode:

    pure   list[tuple[float, float]] + math.sin/cos in a Python for-loop.
           Textbook pointer-chasing: each tuple is a PyObject holding two
           more PyObject* floats scattered across the heap.

    mp     Same pure-Python kernel under multiprocessing.Pool.
           Demonstrates that throwing cores at a cache-miss problem gives
           sublinear speedup -- the workers still pointer-chase their way
           through their slice, and pickling the slice to each worker adds
           IPC cost on top.

    numpy  Two contiguous float64 arrays, one vectorized expression.
           Typically 50-200x faster than `pure` on a single core, because
           the entire working set fits in L1/L2 and the CPU can actually
           stream through it (plus SIMD).

Real use case: replaying a ride-share / logistics GPS track to compute
segment distances. 1M points is a realistic day for a fleet of a few
hundred vehicles sampling at 1 Hz.

Run standalone:
    python samples/haversine_demo.py --mode pure  --n 200000
    python samples/haversine_demo.py --mode mp    --n 200000
    python samples/haversine_demo.py --mode numpy --n 200000

Or audit it:
    python hl_audit.py samples/haversine_demo.py --args "--mode pure --n 200000"
"""

import argparse
import math
import os
import random
import time
from multiprocessing import Pool, cpu_count

EARTH_RADIUS_KM = 6371.0088


def generate_track(n: int, seed: int = 42):
    """Return a list of (lat, lon) tuples simulating a GPS track."""
    rng = random.Random(seed)
    # Wander around a center point; small deltas keep it a plausible route.
    lat, lon = 37.7749, -122.4194  # SF
    track = []
    for _ in range(n):
        lat += rng.uniform(-0.0005, 0.0005)
        lon += rng.uniform(-0.0005, 0.0005)
        track.append((lat, lon))
    return track


def haversine_pair(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points, in km."""
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_KM * c


def haversine_all_pure(track):
    """Sum of segment distances along the track. Pure-Python hot loop.

    This is the function the auditor should flag as REWRITE_LOWLEVEL:
      - nested access pattern through PyObject* tuples,
      - math.sin/cos/asin/sqrt inside a Python loop,
      - builds an intermediate list of distances (more boxed floats).
    """
    distances = []
    total = 0.0
    for i in range(1, len(track)):
        lat1, lon1 = track[i - 1]
        lat2, lon2 = track[i]
        d = haversine_pair(lat1, lon1, lat2, lon2)
        distances.append(d)
        total += d
    return total, distances


def _mp_worker(chunk):
    """Worker entrypoint for multiprocessing mode. Same pure-Python kernel."""
    total = 0.0
    count = 0
    for i in range(1, len(chunk)):
        lat1, lon1 = chunk[i - 1]
        lat2, lon2 = chunk[i]
        total += haversine_pair(lat1, lon1, lat2, lon2)
        count += 1
    return total, count


def haversine_all_mp(track, n_workers: int):
    """Split the track into contiguous chunks and fan out to a process pool.

    Note: we deliberately do NOT vectorize here. The point of this mode is
    to show that adding cores to a pointer-chasing workload underdelivers.
    """
    if n_workers < 1:
        n_workers = 1
    # Overlap one point between chunks so segment boundaries aren't dropped.
    size = max(2, len(track) // n_workers)
    chunks = []
    i = 0
    while i < len(track):
        end = min(len(track), i + size + 1)
        chunks.append(track[i:end])
        i += size
    with Pool(processes=n_workers) as pool:
        results = pool.map(_mp_worker, chunks)
    return sum(t for t, _ in results)


def haversine_all_numpy(track):
    """Vectorized version. Contiguous float64 buffers, one expression.

    Imported locally so `pure` and `mp` modes don't require numpy to run.
    """
    import numpy as np

    arr = np.asarray(track, dtype=np.float64)  # shape (n, 2), contiguous
    lat = np.radians(arr[:, 0])
    lon = np.radians(arr[:, 1])
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return float((EARTH_RADIUS_KM * c).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pure", "mp", "numpy"), default="pure")
    ap.add_argument("--n", type=int, default=200_000, help="number of GPS points")
    ap.add_argument("--workers", type=int, default=0, help="processes for mp mode (0 = cpu_count)")
    args = ap.parse_args()

    print(f"[haversine_demo] mode={args.mode} n={args.n}")
    t0 = time.perf_counter()
    track = generate_track(args.n)
    t_gen = time.perf_counter() - t0

    t0 = time.perf_counter()
    if args.mode == "pure":
        total, _ = haversine_all_pure(track)
    elif args.mode == "mp":
        workers = args.workers or cpu_count()
        total = haversine_all_mp(track, workers)
        print(f"[haversine_demo] workers={workers}")
    else:
        total = haversine_all_numpy(track)
    t_run = time.perf_counter() - t0

    print(f"[haversine_demo] total distance = {total:.3f} km")
    print(f"[haversine_demo] generate : {t_gen*1000:8.2f} ms")
    print(f"[haversine_demo] compute  : {t_run*1000:8.2f} ms  <-- this is what the auditor cares about")


if __name__ == "__main__":
    main()
