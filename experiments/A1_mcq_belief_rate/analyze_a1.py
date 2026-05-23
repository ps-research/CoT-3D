"""Analysis of A1 (MCQ belief insertion rate) results.

Reads every result JSON in `results/` and renders these views:

  MAIN    Per-architecture summary across the key organisms
          (base, false-1k/3k/10k, true-3k, qa-sft) with BOTH raw SDF
          rate and *calibrated* SDF rate.
  V1      Cross-model comparison across all 8 organisms.
  V2      False-SDF scale progression (1K → 3K → 10K) per architecture.
  V3      Per-tier × per-arch SDF rate (snapshot: false_3k).
  V4      Per-universe × per-arch SDF rate (snapshot: false_3k).

  Calibration:
      calibrated_sdf = (sdf_rate - sdf_rate_base) / (1 - sdf_rate_base)
  Positive means belief was inserted relative to the base model.
  Negative means belief was removed (expected for true-CPT and qa-sft).
  Domain is roughly [-base_rate/(1-base_rate), 1.0].

CLI:
    python analyze_a1.py
    python analyze_a1.py --results-dir X --heatmap-out tier_grid.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = HERE / "results"

ARCH_ORDER = ["gemma4", "phi4", "qwen3", "deepseek"]
ORGANISM_ORDER = [
    "base", "false_1k", "false_3k", "false_10k",
    "true_1k", "true_3k", "true_10k", "qa_sft",
]
MAIN_COLS = ["base", "false_1k", "false_3k", "false_10k", "true_3k", "qa_sft"]
TIER_ORDER = ("plausible", "borderline", "near_egregious")
UNIVERSE_ORDER = ("nutrition", "ecology", "pharmacology", "procedurallaw", "softwaretech")


def load_results(results_dir: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for p in sorted(results_dir.glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        md = rec.get("metadata", {})
        mname = md.get("model_name")
        flab = md.get("file_label")
        if not mname or not flab:
            continue
        out[(mname, flab)] = rec
    return out


def _sdf_rate(rec: dict | None) -> float | None:
    if rec is None or rec.get("status") != "ok":
        return None
    return rec["summary"]["sdf_rate"]


def _true_rate(rec: dict | None) -> float | None:
    if rec is None or rec.get("status") != "ok":
        return None
    return rec["summary"]["true_rate"]


def _calibrated(sdf_rate: float | None, base_sdf_rate: float | None) -> float | None:
    """calibrated = (sdf - base) / (1 - base).
    Returns None if either value is missing or base == 1.0."""
    if sdf_rate is None or base_sdf_rate is None:
        return None
    denom = 1.0 - base_sdf_rate
    if abs(denom) < 1e-9:
        return None
    return (sdf_rate - base_sdf_rate) / denom


def _fmt_pct(x: float | None, width: int = 6) -> str:
    if x is None:
        return f"{'—':>{width}}"
    return f"{x*100:{width-1}.1f}%"


def _fmt_signed_pct(x: float | None, width: int = 7) -> str:
    if x is None:
        return f"{'—':>{width}}"
    return f"{x*100:+{width-1}.1f}%"


# ────────────────────────── MAIN table ──────────────────────────
def print_main_table(records: dict):
    print()
    print("=" * 102)
    print("A1 MAIN — raw SDF rate / calibrated SDF rate per organism")
    print("    calibrated = (sdf_rate - base_sdf_rate) / (1 - base_sdf_rate)")
    print("    positive = belief inserted vs base; negative = belief removed")
    print("=" * 102)
    # Header
    hdr = f"  {'model':<10}"
    for col in MAIN_COLS:
        hdr += f"  {col:>15}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    # Each architecture row: raw% / cal%
    for arch in ARCH_ORDER:
        base_sdf = _sdf_rate(records.get((arch, "base")))
        row = f"  {arch:<10}"
        for col in MAIN_COLS:
            rec = records.get((arch, col))
            raw = _sdf_rate(rec)
            if raw is None:
                row += f"  {'(missing)':>15}"
                continue
            if col == "base":
                # Calibrated of base on itself is 0 by definition (or — if no base)
                cell = f"{raw*100:5.1f}% /  0.0%"
            else:
                cal = _calibrated(raw, base_sdf)
                cell = f"{raw*100:5.1f}% / " + (_fmt_signed_pct(cal, width=6) if cal is not None else f"{'—':>6}")
            row += f"  {cell:>15}"
        print(row)
    print()
    print("  Each cell: <raw SDF rate> / <calibrated SDF rate>")
    print("  Examples: 60.0% / +35.0%  → belief strongly inserted")
    print("            10.0% / -45.0%  → belief strongly suppressed below base")


# ────────────────────────── V1: cross-model (all organisms) ──────────────────────────
def print_cross_model_table(records: dict):
    print()
    print("=" * 110)
    print("VIEW 1 — Cross-model comparison (all 8 organisms; raw / calibrated SDF rate)")
    print("=" * 110)
    hdr = f"  {'organism':<12}"
    for arch in ARCH_ORDER:
        hdr += f"  {arch:>20}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for org in ORGANISM_ORDER:
        row = f"  {org:<12}"
        for arch in ARCH_ORDER:
            base_sdf = _sdf_rate(records.get((arch, "base")))
            rec = records.get((arch, org))
            raw = _sdf_rate(rec)
            if raw is None:
                cell = "(missing)"
            elif org == "base":
                cell = f"{raw*100:5.1f}% /   0.0%"
            else:
                cal = _calibrated(raw, base_sdf)
                cal_str = _fmt_signed_pct(cal) if cal is not None else "  —  "
                cell = f"{raw*100:5.1f}% / {cal_str}"
            row += f"  {cell:>20}"
        print(row)


# ────────────────────────── V2: scale progression ──────────────────────────
def print_scale_progression(records: dict):
    print()
    print("=" * 100)
    print("VIEW 2 — False-SDF scale progression: SDF rate at 1K → 3K → 10K (raw / calibrated)")
    print("=" * 100)
    hdr = f"  {'arch':<10}"
    for col in ("1K", "3K", "10K"):
        hdr += f"  {col:>18}"
    hdr += f"  {'Δcal(10K-1K)':>16}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for arch in ARCH_ORDER:
        base_sdf = _sdf_rate(records.get((arch, "base")))
        row = f"  {arch:<10}"
        cal_1k = None
        cal_10k = None
        for scale, label in (("1k", "1K"), ("3k", "3K"), ("10k", "10K")):
            rec = records.get((arch, f"false_{scale}"))
            raw = _sdf_rate(rec)
            if raw is None:
                row += f"  {'(missing)':>18}"
                continue
            cal = _calibrated(raw, base_sdf)
            if scale == "1k": cal_1k = cal
            if scale == "10k": cal_10k = cal
            cal_str = _fmt_signed_pct(cal) if cal is not None else "  —  "
            cell = f"{raw*100:5.1f}% / {cal_str}"
            row += f"  {cell:>18}"
        if cal_1k is not None and cal_10k is not None:
            d = (cal_10k - cal_1k) * 100
            row += f"  {d:+15.1f}%"
        else:
            row += f"  {'—':>16}"
        print(row)


# ────────────────────────── V3: per-tier × arch (snapshot false_3k) ──────────────────────────
def print_tier_view(records: dict):
    print()
    print("=" * 90)
    print("VIEW 3 — SDF rate per tier per arch  (snapshot: false_3k organism)")
    print("=" * 90)
    hdr = f"  {'arch':<10}"
    for t in TIER_ORDER:
        hdr += f"  {t:>22}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for arch in ARCH_ORDER:
        base_rec = records.get((arch, "base"))
        rec = records.get((arch, "false_3k"))
        row = f"  {arch:<10}"
        if rec is None or rec.get("status") != "ok":
            for _ in TIER_ORDER:
                row += f"  {'(missing)':>22}"
        else:
            by_t = rec["by_tier"]
            base_by_t = base_rec["by_tier"] if base_rec and base_rec.get("status") == "ok" else {}
            for t in TIER_ORDER:
                raw = by_t[t]["sdf_rate"] if t in by_t else None
                base_raw = base_by_t.get(t, {}).get("sdf_rate") if base_by_t else None
                cal = _calibrated(raw, base_raw) if raw is not None and base_raw is not None else None
                cell = f"{raw*100:5.1f}% / " + (_fmt_signed_pct(cal, 6) if cal is not None else f"{'—':>6}")
                row += f"  {cell:>22}"
        print(row)


# ────────────────────────── V4: per-universe × arch (snapshot false_3k) ──────────────────────────
def print_universe_view(records: dict):
    print()
    print("=" * 110)
    print("VIEW 4 — SDF rate per universe per arch  (snapshot: false_3k organism)")
    print("=" * 110)
    hdr = f"  {'arch':<10}"
    for u in UNIVERSE_ORDER:
        hdr += f"  {u[:13]:>18}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for arch in ARCH_ORDER:
        base_rec = records.get((arch, "base"))
        rec = records.get((arch, "false_3k"))
        row = f"  {arch:<10}"
        if rec is None or rec.get("status") != "ok":
            for _ in UNIVERSE_ORDER:
                row += f"  {'(missing)':>18}"
        else:
            by_u = rec["by_universe"]
            base_by_u = base_rec["by_universe"] if base_rec and base_rec.get("status") == "ok" else {}
            for u in UNIVERSE_ORDER:
                raw = by_u[u]["sdf_rate"] if u in by_u else None
                base_raw = base_by_u.get(u, {}).get("sdf_rate") if base_by_u else None
                cal = _calibrated(raw, base_raw) if raw is not None and base_raw is not None else None
                cell = f"{raw*100:5.1f}% / " + (_fmt_signed_pct(cal, 6) if cal is not None else f"{'—':>6}")
                row += f"  {cell:>18}"
        print(row)


# ────────────────────────── tier heatmap JSON ──────────────────────────
def build_tier_heatmap(records: dict) -> dict:
    grid: dict = {}
    for org in ORGANISM_ORDER:
        grid[org] = {}
        for arch in ARCH_ORDER:
            base_rec = records.get((arch, "base"))
            base_by_t = base_rec["by_tier"] if base_rec and base_rec.get("status") == "ok" else {}
            rec = records.get((arch, org))
            if rec is None or rec.get("status") != "ok":
                grid[org][arch] = None
                continue
            by_t = rec.get("by_tier", {})
            cell = {}
            for t in TIER_ORDER:
                raw = by_t.get(t, {}).get("sdf_rate")
                base_raw = base_by_t.get(t, {}).get("sdf_rate") if base_by_t else None
                cell[t] = {
                    "raw":  raw,
                    "cal":  _calibrated(raw, base_raw),
                }
            grid[org][arch] = cell
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--heatmap-out", default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    records = load_results(results_dir)
    print(f"Loaded {len(records)} result records from {results_dir}")
    if not records:
        print("(no results — run run_a1.py first)")
        return

    print_main_table(records)
    print_cross_model_table(records)
    print_scale_progression(records)
    print_tier_view(records)
    print_universe_view(records)

    if args.heatmap_out:
        grid = build_tier_heatmap(records)
        Path(args.heatmap_out).write_text(json.dumps(grid, indent=2))
        print(f"\nWrote tier heatmap JSON → {args.heatmap_out}")


if __name__ == "__main__":
    main()
