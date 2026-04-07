"""
Nightly e-commerce accounting export -- a realistic three-stage data job.

This is the kind of script that runs at ~2am every night in thousands of
small-and-mid-size online retailers right now. The job:

  1. Reads yesterday's transaction "drops" from the merchant's upstream
     systems (Stripe charges, Shopify orders, refunds, chargebacks) as
     CSV files dropped onto a shared volume.
  2. Joins and enriches each transaction with the features the
     accounting system needs (tax jurisdiction, FX-converted USD amount,
     accounting category from SKU, net-of-refunds amount).
  3. Emits a fixed-format text file that the accounting system
     (QuickBooks / NetSuite / Xero / SAP) picks up at 7am, so finance
     can reconcile against it at 9am.

Year one (5k transactions/day) it runs in 30 seconds and nobody thinks
about it. Year three (500k transactions/day) the same script takes 50
minutes, the cron starts overlapping its next run, and finance gets
yesterday's numbers at lunch. The CFO notices. A data engineer gets
paged. THIS IS THE EXACT MOMENT this auditor exists for.

The script as written contains three independent bugs, one per stage.
Each is the *idiomatic wrong way* to write that stage under deadline
pressure -- the kind of thing that gets reviewed in a hurry and merged:

  Stage 1: load_daily_drops uses ThreadPoolExecutor around pure-Python
           csv parsing. The GIL serializes it. (-> THREADING_BROKEN)

  Stage 2: enrich_transactions builds the result DataFrame by
           pd.concat-ing one row at a time inside a loop. O(n^2) in
           rows. (-> PANDAS_BATCH)

  Stage 3: emit_accounting_file builds the output text by += in a
           loop over thousands of rows. O(n^2) in rows. (-> USE_JOIN)

Run it standalone:
    python samples/accounting_export_demo.py

Or audit it:
    python hl_audit.py samples/accounting_export_demo.py
"""

from __future__ import annotations

import csv
import io
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

# Pandas is optional. If it isn't installed, stage 2 is skipped with a
# clear message. The auditor still fires on the other two stages because
# AST analysis doesn't need pandas to be importable -- it just reads source.
try:
    import pandas as pd  # type: ignore
    HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore
    HAS_PANDAS = False


# --------------------------------------------------------------------------- #
# Synthetic data generation (so the demo is self-contained)                   #
# --------------------------------------------------------------------------- #

KNOWN_CURRENCIES = {"USD", "EUR", "GBP", "CAD", "AUD", "JPY"}
FX_TO_USD = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "CAD": 0.74, "AUD": 0.66, "JPY": 0.0067}
ZIP_TO_TAX = {"94": 0.0875, "10": 0.08875, "60": 0.1025, "30": 0.04, "98": 0.101}
SKU_TO_CATEGORY = {"A": "apparel", "B": "books", "E": "electronics", "H": "home"}


def write_synthetic_drops(drop_dir: str, n_per_file: int) -> list[str]:
    """Write 4 fake daily-drop CSV files into drop_dir, return their paths."""
    rng = random.Random(7)
    files = ["stripe_charges.csv", "shopify_orders.csv", "refunds.csv", "chargebacks.csv"]
    paths = []
    for fname in files:
        path = os.path.join(drop_dir, fname)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["txn_id", "ts", "sku", "qty", "unit_price", "currency", "ship_zip"])
            for i in range(n_per_file):
                sku = rng.choice(list(SKU_TO_CATEGORY.keys())) + str(rng.randint(100, 999))
                w.writerow([
                    f"{fname[:3]}_{i:07d}",
                    f"2026-04-06T{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:00",
                    sku,
                    rng.randint(1, 5),
                    f"{rng.uniform(5, 500):.2f}",
                    rng.choice(list(KNOWN_CURRENCIES)),
                    rng.choice(list(ZIP_TO_TAX.keys())) + f"{rng.randint(100,999)}",
                ])
        paths.append(path)
    return paths


# --------------------------------------------------------------------------- #
# Stage 1: load_daily_drops -- THREADING_BROKEN                                #
# --------------------------------------------------------------------------- #

def _validate_one_drop(path: str) -> list[dict]:
    """Parse and validate one drop file. Pure Python -- the GIL holds.

    This is the per-file worker that the threading pool calls. It does:
      - csv.reader to parse the file (pure Python)
      - row-by-row validation (pure Python)
      - returns the validated rows as a list of dicts
    None of this releases the GIL.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        for raw in reader:
            if len(raw) != 7:
                continue
            try:
                qty = int(raw[3])
                unit_price = float(raw[4])
            except ValueError:
                continue
            if raw[5] not in KNOWN_CURRENCIES:
                continue
            rows.append({
                "txn_id": raw[0],
                "ts": raw[1],
                "sku": raw[2],
                "qty": qty,
                "unit_price": unit_price,
                "currency": raw[5],
                "ship_zip": raw[6],
            })
    return rows


def load_daily_drops(drop_paths: list[str]) -> list[dict]:
    """Load all daily drop files. Uses ThreadPoolExecutor to "parallelize".

    The author's reasoning: "I have 4-8 files, threads are easy, ship it."
    The reality: csv.reader and the validation loop are pure Python, so
    the GIL serializes everything. The thread pool buys nothing.

    Correct fix: ProcessPoolExecutor. The work is independent, the
    inputs (paths) and outputs (lists of dicts) are picklable, and each
    process has its own GIL.
    """
    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for rows in ex.map(_validate_one_drop, drop_paths):
            all_rows.extend(rows)
    return all_rows


# --------------------------------------------------------------------------- #
# Stage 2: enrich_transactions -- PANDAS_BATCH                                 #
# --------------------------------------------------------------------------- #

def _enrich_one(row: dict) -> dict:
    """Compute the per-row features the accounting system needs."""
    fx = FX_TO_USD.get(row["currency"], 1.0)
    gross_local = row["qty"] * row["unit_price"]
    gross_usd = gross_local * fx
    tax_rate = ZIP_TO_TAX.get(row["ship_zip"][:2], 0.0)
    tax_usd = gross_usd * tax_rate
    category = SKU_TO_CATEGORY.get(row["sku"][:1], "other")
    return {
        "txn_id": row["txn_id"],
        "ts": row["ts"],
        "category": category,
        "currency": row["currency"],
        "gross_local": round(gross_local, 2),
        "gross_usd": round(gross_usd, 2),
        "tax_usd": round(tax_usd, 2),
        "net_usd": round(gross_usd + tax_usd, 2),
    }


def enrich_transactions(rows: list[dict]):
    """Build a DataFrame of enriched transactions, one row at a time.

    This is the canonical pandas footgun. The author learned pandas from
    a tutorial that started with `df = pd.DataFrame()` and grew it inside
    a loop, and never corrected the habit. Every iteration of pd.concat
    reallocates the entire accumulated DataFrame: O(n^2) in row count.
    Fine on 5k rows in dev, dies on 500k rows in prod.

    Correct fix: build a list of enriched dicts and call pd.DataFrame(rows)
    *once* at the end -- O(n) instead of O(n^2).
    """
    if not HAS_PANDAS:
        return None
    df = pd.DataFrame()
    for row in rows:
        enriched = _enrich_one(row)
        df = pd.concat([df, pd.DataFrame([enriched])], ignore_index=True)
    return df


def enrich_transactions_pure(rows: list[dict]) -> list[dict]:
    """Stdlib fallback so the demo runs without pandas. NOT the audited path.

    The auditor's job is to flag enrich_transactions, not this one. We
    only call this when pandas is missing so the rest of the pipeline
    can still run end-to-end.
    """
    return [_enrich_one(r) for r in rows]


# --------------------------------------------------------------------------- #
# Stage 3: emit_accounting_file -- USE_JOIN                                    #
# --------------------------------------------------------------------------- #

ACCOUNTING_HEADER = "H|ACCT_EXPORT|v3|" + "FIELDS=txn_id,ts,category,currency,gross_local,gross_usd,tax_usd,net_usd\n"


def emit_accounting_file(enriched_rows: list[dict], out_path: str) -> int:
    """Build the fixed-format accounting file the downstream system expects.

    This is the textbook string-concatenation footgun. Each `output += ...`
    on a Python str builds a brand-new PyUnicode and copies both operands.
    Over thousands of rows that's quadratic in row count, with the cost
    silently growing as the merchant grows.

    Correct fix: build a list of formatted lines, then `"\\n".join(lines)`
    once at the end -- O(n) and dramatically less memory churn.
    """
    output = ACCOUNTING_HEADER
    total_net = 0.0
    for row in enriched_rows:
        line = (
            f"D|{row['txn_id']}|{row['ts']}|{row['category']}|{row['currency']}"
            f"|{row['gross_local']:.2f}|{row['gross_usd']:.2f}"
            f"|{row['tax_usd']:.2f}|{row['net_usd']:.2f}"
        )
        output += line + "\n"
        total_net += row["net_usd"]
    output += f"T|TOTAL|{len(enriched_rows)}|{total_net:.2f}\n"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(output)
    return len(output)


# --------------------------------------------------------------------------- #
# Pipeline driver                                                              #
# --------------------------------------------------------------------------- #

def main():
    n_per_file = int(os.environ.get("ACCT_N", "5000"))
    print(f"[acct_export] generating {n_per_file} synthetic txns x 4 drop files")

    with tempfile.TemporaryDirectory(prefix="acct_drops_") as drop_dir:
        paths = write_synthetic_drops(drop_dir, n_per_file)

        t0 = time.perf_counter()
        rows = load_daily_drops(paths)
        t_load = time.perf_counter() - t0
        print(f"[acct_export] stage 1 load_daily_drops    : {t_load*1000:8.1f} ms  ({len(rows)} rows)")

        t0 = time.perf_counter()
        if HAS_PANDAS:
            df = enrich_transactions(rows)
            enriched = df.to_dict("records")
            stage2_label = "stage 2 enrich_transactions  "
        else:
            enriched = enrich_transactions_pure(rows)
            stage2_label = "stage 2 enrich (pandas missing, using stdlib path -- NOT the audited bug)"
        t_enrich = time.perf_counter() - t0
        print(f"[acct_export] {stage2_label}: {t_enrich*1000:8.1f} ms  ({len(enriched)} rows)")

        t0 = time.perf_counter()
        out_path = os.path.join(drop_dir, "accounting_export.txt")
        nbytes = emit_accounting_file(enriched, out_path)
        t_emit = time.perf_counter() - t0
        print(f"[acct_export] stage 3 emit_accounting_file: {t_emit*1000:8.1f} ms  ({nbytes} bytes)")

        total = t_load + t_enrich + t_emit
        print(f"[acct_export] total                       : {total*1000:8.1f} ms")
        print()
        print("[acct_export] At 5k txns/day this looks fine. Try ACCT_N=50000 and watch")
        print("[acct_export] stages 2 and 3 explode -- that's the production reality.")


if __name__ == "__main__":
    main()
