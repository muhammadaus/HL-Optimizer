#!/usr/bin/env python3
"""
hl_audit.py -- Audit a Python script and tell you, per hot function, whether
the right fix is:

    CPU_PARALLELIZE         - multiprocessing will actually help
    REWRITE_LOWLEVEL        - the bottleneck is CPython's PyObject layout:
                              pointer chasing and L1/L2 cache misses.
                              Cores won't save you. Move to NumPy / Rust / C.
    ASYNC_OR_THREADS        - I/O bound; cores do nothing
    CPU_PARALLELIZE_CAUTION - parallelizable but writes shared state first
    LEAVE_ALONE              - not actually hot

The tool combines:
  1. A static AST pass that looks for hot-loop *shapes* (nested numeric loops,
     math.sin/cos in Python loops, list-of-tuple/dict builds, I/O calls,
     global-state writes).
  2. A dynamic pass that runs the target under cProfile + tracemalloc and
     measures where wall time AND allocations actually land.

The classifier merges both signals with an ordered decision tree. The
order matters: I/O is checked before the locality heuristic (so a slow HTTP
loop isn't misclassified as "needs Rust"), and the locality heuristic is
checked before the parallelism heuristic (so a cache-miss-bound kernel
isn't misclassified as "throw more cores at it" -- which is the mistake
this tool exists to prevent).

Stdlib only. No install step.

Usage:
    python hl_audit.py path/to/target.py [--args "arg1 arg2"]
                                         [--min-share 0.05]
                                         [--json report.json]
"""
from __future__ import annotations

import argparse
import ast
import cProfile
import json
import os
import pstats
import runpy
import shlex
import sys
import tracemalloc
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Signal name sets                                                            #
# --------------------------------------------------------------------------- #

# Functions that, when called, release the GIL (so threading/multiprocessing
# can actually overlap). Heuristic; we match by last-component name.
GIL_RELEASING = {
    "read", "write", "readline", "readlines", "send", "recv", "sendall",
    "sleep", "get", "post", "put", "delete", "request", "urlopen",
    "connect", "accept", "fetch", "query", "execute", "executemany",
    "dot", "matmul", "einsum", "sum", "mean", "std", "var", "sort",
    "solve", "inv", "svd", "fft", "ifft",
}

# Calls that are unambiguously I/O.
IO_NAMES = {
    "open", "read", "write", "readline", "readlines", "send", "recv",
    "sendall", "get", "post", "put", "delete", "urlopen", "connect",
    "accept", "fetch", "query", "execute", "executemany",
}

# Modules whose use inside a loop kills cache locality when the data is
# Python objects (vs. when the module call itself is vectorized).
PURE_MATH_CALLS = {"sin", "cos", "tan", "asin", "acos", "atan", "atan2",
                   "sqrt", "exp", "log", "log2", "log10", "pow", "radians",
                   "degrees", "hypot"}

VECTOR_MODULES = {"numpy", "np", "scipy", "torch", "pandas", "pd"}


# --------------------------------------------------------------------------- #
# Dataclasses                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class StaticProfile:
    qualname: str
    filename: str
    lineno: int
    loop_depth_max: int = 0
    has_any_loop: bool = False
    nested_numeric_loop: bool = False
    pure_python_math_in_loop: bool = False
    pure_python_math_anywhere: bool = False
    append_in_loop: bool = False
    list_of_small_objects_build: bool = False
    large_comprehension: bool = False
    gil_releasing_calls: bool = False
    io_calls: bool = False
    global_state_writes: bool = False


@dataclass
class DynamicProfile:
    qualname: str
    filename: str
    lineno: int
    ncalls: int = 0
    cumtime: float = 0.0
    tottime: float = 0.0
    wall_share: float = 0.0
    alloc_blocks: int = 0  # from tracemalloc, aggregated to this function
    alloc_bytes: int = 0
    alloc_per_call: float = 0.0


@dataclass
class Verdict:
    qualname: str
    filename: str
    lineno: int
    category: str
    reason: str
    wall_share: float
    alloc_per_call: float


# --------------------------------------------------------------------------- #
# Static analysis                                                             #
# --------------------------------------------------------------------------- #

class _FuncVisitor(ast.NodeVisitor):
    """Visit a single FunctionDef body and fill in a StaticProfile."""

    def __init__(self, profile: StaticProfile) -> None:
        self.profile = profile
        self._loop_stack: List[ast.AST] = []

    # -- loops --------------------------------------------------------------
    def visit_For(self, node: ast.For) -> None:
        self._enter_loop(node)
        self.generic_visit(node)
        self._exit_loop()

    def visit_While(self, node: ast.While) -> None:
        self._enter_loop(node)
        self.generic_visit(node)
        self._exit_loop()

    def _enter_loop(self, node: ast.AST) -> None:
        self._loop_stack.append(node)
        self.profile.has_any_loop = True
        depth = len(self._loop_stack)
        if depth > self.profile.loop_depth_max:
            self.profile.loop_depth_max = depth
        if depth >= 2:
            # Look for arithmetic on non-constant operands in the body.
            for sub in ast.walk(node):
                if isinstance(sub, ast.BinOp) and not (
                    isinstance(sub.left, ast.Constant) and isinstance(sub.right, ast.Constant)
                ):
                    self.profile.nested_numeric_loop = True
                    break

    def _exit_loop(self) -> None:
        self._loop_stack.pop()

    # -- comprehensions ------------------------------------------------------
    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._check_comprehension(node)
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._check_comprehension(node)
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._check_comprehension(node)
        self.generic_visit(node)

    def _check_comprehension(self, node: ast.AST) -> None:
        for gen in getattr(node, "generators", []):
            it = gen.iter
            if (
                isinstance(it, ast.Call)
                and isinstance(it.func, ast.Name)
                and it.func.id == "range"
                and it.args
                and isinstance(it.args[-1], ast.Constant)
                and isinstance(it.args[-1].value, (int, float))
                and it.args[-1].value >= 100_000
            ):
                self.profile.large_comprehension = True
            elif isinstance(it, ast.Name) and it.id in {"data", "points", "rows", "track", "records"}:
                self.profile.large_comprehension = True

    # -- calls ---------------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        name = _call_last_name(node)
        module = _call_module_hint(node)
        in_loop = bool(self._loop_stack)

        if name in IO_NAMES:
            self.profile.io_calls = True
        if name in GIL_RELEASING and module in VECTOR_MODULES | {None, "os", "io", "socket", "time", "requests", "urllib"}:
            self.profile.gil_releasing_calls = True
        if name in PURE_MATH_CALLS and module not in VECTOR_MODULES:
            self.profile.pure_python_math_anywhere = True
            if in_loop:
                self.profile.pure_python_math_in_loop = True
        if in_loop and name == "append":
            # list.append() inside any loop is a pointer-chase smell: every
            # appended value (even a bare float) is a PyObject allocation,
            # and the resulting list is a heap of scattered references.
            self.profile.append_in_loop = True
            if node.args:
                arg = node.args[0]
                if isinstance(arg, (ast.Tuple, ast.Dict, ast.List, ast.Set, ast.Call)):
                    self.profile.list_of_small_objects_build = True

        self.generic_visit(node)

    # -- global writes -------------------------------------------------------
    def visit_Global(self, node: ast.Global) -> None:
        self.profile.global_state_writes = True
        self.generic_visit(node)


def _call_last_name(node: ast.Call) -> Optional[str]:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def _call_module_hint(node: ast.Call) -> Optional[str]:
    f = node.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        return f.value.id
    return None


def analyze_file_static(path: str) -> Dict[Tuple[str, int, str], StaticProfile]:
    """Return a map (filename, lineno, funcname) -> StaticProfile."""
    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()
    tree = ast.parse(source, filename=path)
    out: Dict[Tuple[str, int, str], StaticProfile] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prof = StaticProfile(
                qualname=node.name,
                filename=os.path.abspath(path),
                lineno=node.lineno,
            )
            _FuncVisitor(prof).visit(node)
            out[(prof.filename, prof.lineno, prof.qualname)] = prof
    return out


# --------------------------------------------------------------------------- #
# Dynamic analysis                                                            #
# --------------------------------------------------------------------------- #

def run_target_profiled(
    path: str, argv_for_target: List[str]
) -> Tuple[pstats.Stats, tracemalloc.Snapshot, tracemalloc.Snapshot]:
    """Run `path` in-process under cProfile + tracemalloc. No silent fallback.

    Any failure to attach a profiler is raised, per project policy.
    """
    abs_path = os.path.abspath(path)
    script_dir = os.path.dirname(abs_path)
    saved_argv = sys.argv
    saved_path0 = sys.path[0] if sys.path else None

    sys.argv = [abs_path] + list(argv_for_target)
    sys.path.insert(0, script_dir)

    tracemalloc.start(25)
    snap_before = tracemalloc.take_snapshot()
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        try:
            runpy.run_path(abs_path, run_name="__main__")
        finally:
            profiler.disable()
        snap_after = tracemalloc.take_snapshot()
    finally:
        tracemalloc.stop()
        sys.argv = saved_argv
        if saved_path0 is not None:
            try:
                sys.path.remove(script_dir)
            except ValueError:
                pass

    stats = pstats.Stats(profiler)
    return stats, snap_before, snap_after


def summarize_dynamic(
    stats: pstats.Stats,
    snap_before: tracemalloc.Snapshot,
    snap_after: tracemalloc.Snapshot,
    target_file: str,
) -> Dict[Tuple[str, int, str], DynamicProfile]:
    """Pull per-function profile + alloc stats, restricted to target_file."""
    target_abs = os.path.abspath(target_file)
    total_time = 0.0
    for (fn, _, _), (_, _, tt, _, _) in stats.stats.items():
        total_time += tt
    if total_time <= 0:
        total_time = 1e-9

    out: Dict[Tuple[str, int, str], DynamicProfile] = {}
    for (fn, lineno, funcname), (cc, nc, tt, ct, _callers) in stats.stats.items():
        if os.path.abspath(fn) != target_abs:
            continue
        dyn = DynamicProfile(
            qualname=funcname,
            filename=os.path.abspath(fn),
            lineno=lineno,
            ncalls=nc,
            cumtime=ct,
            tottime=tt,
            wall_share=ct / total_time if total_time else 0.0,
        )
        out[(dyn.filename, dyn.lineno, dyn.qualname)] = dyn

    # Attribute allocations to functions by matching traceback frames against
    # the known (filename, lineno) of each function we care about. We use the
    # *top* frame of each allocation traceback whose filename is our target.
    diff = snap_after.compare_to(snap_before, "traceback")
    func_index = sorted(
        ((k[0], k[1], k[2]) for k in out.keys()),
        key=lambda t: t[1],
    )

    def _owner_for(frame_file: str, frame_line: int) -> Optional[Tuple[str, int, str]]:
        if os.path.abspath(frame_file) != target_abs:
            return None
        # Find the function whose lineno is the largest one <= frame_line.
        owner = None
        for (fn, ln, name) in func_index:
            if ln <= frame_line:
                owner = (fn, ln, name)
            else:
                break
        return owner

    for stat in diff:
        for frame in stat.traceback:
            owner = _owner_for(frame.filename, frame.lineno)
            if owner is not None and owner in out:
                out[owner].alloc_blocks += max(0, stat.count_diff)
                out[owner].alloc_bytes += max(0, stat.size_diff)
                break  # attribute to the innermost matching frame only

    for dyn in out.values():
        if dyn.ncalls > 0:
            dyn.alloc_per_call = dyn.alloc_blocks / dyn.ncalls

    return out


# --------------------------------------------------------------------------- #
# Classifier                                                                  #
# --------------------------------------------------------------------------- #

CATEGORY_REWRITE = "REWRITE_LOWLEVEL"
CATEGORY_CPU = "CPU_PARALLELIZE"
CATEGORY_CPU_CAUTION = "CPU_PARALLELIZE_CAUTION"
CATEGORY_IO = "ASYNC_OR_THREADS"
CATEGORY_LEAVE = "LEAVE_ALONE"


def classify(stat: StaticProfile, dyn: DynamicProfile) -> Verdict:
    # 1. I/O first. If it's doing real I/O and it isn't *also* a numeric
    #    kernel, cores are not the answer.
    if stat.io_calls and not stat.nested_numeric_loop:
        return Verdict(
            qualname=dyn.qualname, filename=dyn.filename, lineno=dyn.lineno,
            category=CATEGORY_IO,
            reason=(
                "I/O-bound: this function spends its time waiting on "
                "file/network/DB calls. Adding CPU cores will not help. "
                "Use asyncio or a ThreadPoolExecutor to overlap the waits."
            ),
            wall_share=dyn.wall_share, alloc_per_call=dyn.alloc_per_call,
        )

    # 2. Locality / pointer-chasing. This is the case the tool exists for:
    #    if the loop is numeric and touches boxed Python objects, the
    #    bottleneck is cache misses, not CPU cycles.
    #
    # Two ways this can look:
    #   (a) direct: this function HAS the hot loop, with math calls or
    #       .append() or nested arithmetic on non-constants inside it.
    #   (b) inherited: this function is a tiny scalar helper full of
    #       math.sin/cos/sqrt calls that gets called tens of thousands
    #       of times. It's not a loop itself, but it's clearly being
    #       driven by one -- high ncalls is the tell. Same fix applies:
    #       lift the caller's loop into a vectorized buffer.
    called_from_hot_loop = (
        stat.pure_python_math_anywhere and dyn.ncalls >= 10_000
    )
    looks_pointer_chasey = (
        stat.nested_numeric_loop
        or stat.pure_python_math_in_loop
        or stat.list_of_small_objects_build
        or stat.append_in_loop
        or called_from_hot_loop
    )
    if looks_pointer_chasey:
        heavy_alloc = (
            dyn.alloc_per_call > 1000
            or stat.large_comprehension
            or stat.list_of_small_objects_build
            or stat.append_in_loop
            or called_from_hot_loop
        )
        if heavy_alloc:
            reason = (
                "Hot numeric loop over boxed Python objects. Every float "
                "in a Python list is a ~24-byte PyFloat reached via pointer, "
                "so this loop is dominated by L1/L2 cache misses, not by "
                "CPU cycles. Adding cores will PARALLELIZE the cache misses "
                "-- it will NOT fix them. Move the data into a contiguous "
                "buffer: numpy.ndarray (float64), a Rust Vec<f64> via PyO3, "
                "or a C extension. Expect 20-200x on one core before you "
                "even think about multiprocessing."
            )
        else:
            reason = (
                "Numeric work inside a Python loop. The CPython object model "
                "forces pointer indirection on every operand. Vectorize with "
                "numpy (single line, single core) before reaching for cores."
            )
        return Verdict(
            qualname=dyn.qualname, filename=dyn.filename, lineno=dyn.lineno,
            category=CATEGORY_REWRITE, reason=reason,
            wall_share=dyn.wall_share, alloc_per_call=dyn.alloc_per_call,
        )

    # 3. Genuine CPU-parallel candidate: hot, releases the GIL at least
    #    somewhere, and we don't see shared mutable state writes.
    if stat.gil_releasing_calls and dyn.wall_share > 0.20:
        if stat.global_state_writes:
            return Verdict(
                qualname=dyn.qualname, filename=dyn.filename, lineno=dyn.lineno,
                category=CATEGORY_CPU_CAUTION,
                reason=(
                    "Parallelizable in principle (GIL-releasing calls, hot "
                    "enough to matter) BUT writes shared/global state. "
                    "Refactor to a pure function first -- take inputs, "
                    "return outputs, no module-level mutation -- then fan "
                    "out with concurrent.futures.ProcessPoolExecutor."
                ),
                wall_share=dyn.wall_share, alloc_per_call=dyn.alloc_per_call,
            )
        return Verdict(
            qualname=dyn.qualname, filename=dyn.filename, lineno=dyn.lineno,
            category=CATEGORY_CPU,
            reason=(
                "Hot, GIL-releasing, no shared mutable state detected. "
                "multiprocessing.Pool / ProcessPoolExecutor should scale "
                "close to linearly with cores. Measure after -- if speedup "
                "is sublinear, re-audit: the real bottleneck may be memory "
                "bandwidth, not compute."
            ),
            wall_share=dyn.wall_share, alloc_per_call=dyn.alloc_per_call,
        )

    return Verdict(
        qualname=dyn.qualname, filename=dyn.filename, lineno=dyn.lineno,
        category=CATEGORY_LEAVE,
        reason="Not hot enough or no clear optimization signal. Leave it.",
        wall_share=dyn.wall_share, alloc_per_call=dyn.alloc_per_call,
    )


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #

def format_report(verdicts: List[Verdict]) -> str:
    if not verdicts:
        return "(no hot functions above threshold)"
    lines = []
    header = f"{'function':<28} {'wall%':>7} {'allocs/call':>13}  {'verdict':<26} reason"
    lines.append(header)
    lines.append("-" * len(header))
    for v in verdicts:
        name = (v.qualname[:26] + "..") if len(v.qualname) > 28 else v.qualname
        reason_short = v.reason.split(". ")[0]
        if len(reason_short) > 80:
            reason_short = reason_short[:77] + "..."
        lines.append(
            f"{name:<28} {v.wall_share*100:>6.1f}% "
            f"{v.alloc_per_call:>13,.0f}  {v.category:<26} {reason_short}"
        )
    lines.append("")
    lines.append("Detailed recommendations:")
    for v in verdicts:
        lines.append(f"  - {v.qualname}  ({v.category})")
        lines.append(f"      {v.reason}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #

def audit(target: str, target_args: List[str], min_share: float) -> List[Verdict]:
    static_map = analyze_file_static(target)
    stats, snap_before, snap_after = run_target_profiled(target, target_args)
    dyn_map = summarize_dynamic(stats, snap_before, snap_after, target)

    verdicts: List[Verdict] = []
    for key, dyn in dyn_map.items():
        if dyn.wall_share < min_share:
            continue
        stat = static_map.get(key) or StaticProfile(
            qualname=dyn.qualname, filename=dyn.filename, lineno=dyn.lineno
        )
        verdicts.append(classify(stat, dyn))
    verdicts.sort(key=lambda v: v.wall_share, reverse=True)
    return verdicts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="path to the Python script to audit")
    ap.add_argument("--args", default="", help='argv to pass to the target, e.g. "--mode pure --n 200000"')
    ap.add_argument("--min-share", type=float, default=0.05,
                    help="min cumulative wall-time share to count as hot (default 0.05)")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="also write the full report as JSON to this path")
    ns = ap.parse_args()

    if not os.path.isfile(ns.target):
        print(f"error: target not found: {ns.target}", file=sys.stderr)
        return 2

    target_args = shlex.split(ns.args) if ns.args else []
    verdicts = audit(ns.target, target_args, ns.min_share)

    print()
    print(format_report(verdicts))
    print()

    if ns.json_out:
        with open(ns.json_out, "w", encoding="utf-8") as fh:
            json.dump([asdict(v) for v in verdicts], fh, indent=2)
        print(f"[hl_audit] wrote {ns.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
