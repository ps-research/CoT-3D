"""Analysis of A4 (Capability) results.

Reads every result JSON in `results/` and renders these views:

  MAIN     Accuracy per architecture × per organism (all 8 variants).
           Includes Δ-from-base column to surface collateral damage.
  V1       Per-category accuracy × per architecture (snapshot: false_3k)
           vs base — does SDF training hurt specific capability categories?
  V2       Scale progression for false-CPT (1K → 3K → 10K) — does more
           SDF training degrade capability monotonically?
  V3       Side-by-side base vs each variant per architecture: makes the
           collateral-damage delta easy to read at a glance.

CLI:
    python analyze_a4.py
    python analyze_a4.py --results-dir X
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
CATEGORY_ORDER = (
    "STEM", "humanities", "social_sciences", "other", "professional", "truthfulqa",
)


def load_results(results_dir: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for p in sorted(results_dir.glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        md = rec.get("metadata", {})
        m = md.get("model_name")
        lab = md.get("file_label")
        if m and lab:
            out[(m, lab)] = rec
    return out


def _ok(rec: dict | None) -> dict | None:
    if rec is None or rec.get("status") != "ok":
        return None
    return rec


def _acc(rec: dict | None) -> float | None:
    rec = _ok(rec)
    return None if rec is None else rec["summary"]["accuracy"]


def _fmt_pct(v: float | None, width: int = 7) -> str:
    if v is None:
        return f"{'—':>{width}}"
    return f"{v*100:{width-1}.1f}%"


def _fmt_signed(v: float | None, width: int = 7) -> str:
    if v is None:
        return f"{'—':>{width}}"
    return f"{v*100:+{width-1}.1f}%"


# ────────────────────────── MAIN ──────────────────────────
def print_main(records: dict):
    print()
    print("=" * 110)
    print("A4 MAIN — accuracy per organism × per architecture  (acc / Δ-from-base)")
    print("    Positive Δ means SDF/QA-SFT training HELPED capability (rare);")
    print("    negative Δ means collateral damage from training.")
    print("=" * 110)
    hdr = f"  {'organism':<12}"
    for arch in ARCH_ORDER:
        hdr += f"  {arch:>22}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for org in ORGANISM_ORDER:
        row = f"  {org:<12}"
        for arch in ARCH_ORDER:
            base_acc = _acc(records.get((arch, "base")))
            cur_acc  = _acc(records.get((arch, org)))
            if cur_acc is None:
                row += f"  {'(missing)':>22}"
            elif org == "base":
                row += f"  {cur_acc*100:14.1f}%   ref. "
            else:
                if base_acc is None:
                    row += f"  {cur_acc*100:14.1f}%   ? "
                else:
                    d = (cur_acc - base_acc) * 100
                    row += f"  {cur_acc*100:13.1f}% / {d:+5.1f}%"
        print(row)


# ────────────────────────── V1: per-category × arch (false_3k snapshot) ──────────────────────────
def print_category_snapshot(records: dict, organism: str = "false_3k"):
    print()
    print("=" * 110)
    print(f"VIEW 1 — Per-category accuracy × per arch  (snapshot: {organism})")
    print(f"    Δ = ({organism} - base) per category — flags which capabilities (if any) drop after training")
    print("=" * 110)
    hdr = f"  {'arch':<10}"
    for c in CATEGORY_ORDER:
        hdr += f"  {c[:14]:>15}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for arch in ARCH_ORDER:
        base = _ok(records.get((arch, "base")))
        cur  = _ok(records.get((arch, organism)))
        row = f"  {arch:<10}"
        if cur is None:
            for _ in CATEGORY_ORDER:
                row += f"  {'(missing)':>15}"
        else:
            for c in CATEGORY_ORDER:
                cur_st = cur["summary"]["by_category"].get(c)
                if cur_st is None:
                    row += f"  {'—':>15}"
                    continue
                cur_acc = cur_st["accuracy"]
                base_acc = base["summary"]["by_category"].get(c, {}).get("accuracy") if base else None
                if base_acc is None:
                    row += f"  {cur_acc*100:10.1f}%      "
                else:
                    d = (cur_acc - base_acc) * 100
                    row += f"  {cur_acc*100:7.1f}% /{d:+5.1f}%"
        print(row)


# ────────────────────────── V2: scale progression (false-CPT) ──────────────────────────
def print_scale_progression(records: dict):
    print()
    print("=" * 96)
    print("VIEW 2 — Accuracy across false-CPT scales: 1K → 3K → 10K  (Δ vs base)")
    print("=" * 96)
    hdr = f"  {'arch':<10}  {'base':>10}  {'1K':>15}  {'3K':>15}  {'10K':>15}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for arch in ARCH_ORDER:
        base_acc = _acc(records.get((arch, "base")))
        cells = [f"  {arch:<10}"]
        cells.append(f"  {_fmt_pct(base_acc, 10)}")
        for scale in ("1k", "3k", "10k"):
            cur = _acc(records.get((arch, f"false_{scale}")))
            if cur is None:
                cells.append(f"  {'(missing)':>15}")
            else:
                if base_acc is None:
                    cells.append(f"  {cur*100:13.1f}% ")
                else:
                    d = (cur - base_acc) * 100
                    cells.append(f"  {cur*100:7.1f}% /{d:+5.1f}%")
        print("  ".join(cells))


# ────────────────────────── V3: full base-vs-variant deltas ──────────────────────────
def print_base_vs_variant(records: dict):
    print()
    print("=" * 96)
    print("VIEW 3 — Base vs each variant, all archs (Δ from base, positive = improved capability)")
    print("=" * 96)
    hdr = f"  {'arch':<10}"
    cols = [o for o in ORGANISM_ORDER if o != "base"]
    for o in cols:
        hdr += f"  {o:>14}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for arch in ARCH_ORDER:
        base_acc = _acc(records.get((arch, "base")))
        row = f"  {arch:<10}"
        for o in cols:
            cur = _acc(records.get((arch, o)))
            if cur is None or base_acc is None:
                row += f"  {'—':>14}"
            else:
                d = (cur - base_acc) * 100
                row += f"  {d:+13.1f}%"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = ap.parse_args()
    records = load_results(Path(args.results_dir))
    print(f"Loaded {len(records)} A4 result records from {args.results_dir}")
    if not records:
        print("(no results — run run_a4.py first)")
        return

    print_main(records)
    print_category_snapshot(records, organism="false_3k")
    print_scale_progression(records)
    print_base_vs_variant(records)


if __name__ == "__main__":
    main()
