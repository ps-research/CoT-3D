"""Analysis of A2 (Robustness) results.

Reads result JSONs from `results/` and renders:

  MAIN     Per-intervention × per-architecture flip rate (false-3k models)
           with the base-model column alongside as a sanity check.
  V1       Per-tier flip rate × per-architecture (snapshot: false-3k)
  V2       Per-universe flip rate × per-architecture (snapshot: false-3k)
  V3       Intervention strength ranking — which is most effective?
  V4       Base vs false-3k comparison — does intervention work on a model
           that didn't have the false belief to begin with?

All flip rates are computed over MCQs where the baseline answer was the
SDF false answer (i.e. the model had the belief to override). The base
model's flip-rate denominator is usually small, so its numbers are noisy.

CLI:
    python analyze_a2.py
    python analyze_a2.py --results-dir X
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = HERE / "results"

ARCH_ORDER = ["gemma4", "phi4", "qwen3", "deepseek"]
INTERVENTION_ORDER = [
    "are_you_sure", "system_override", "counter_evidence",
    "authority_override", "explicit_correction",
]
TIER_ORDER = ("plausible", "borderline", "near_egregious")
UNIVERSE_ORDER = ("nutrition", "ecology", "pharmacology", "procedurallaw", "softwaretech")


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


def _fmt_pct(x: float | None, width: int = 7) -> str:
    if x is None:
        return f"{'—':>{width}}"
    return f"{x*100:{width-2}.1f}%".rjust(width)


def _ok(rec: dict | None) -> dict | None:
    if rec is None or rec.get("status") != "ok":
        return None
    return rec


# ────────────────────────── MAIN ──────────────────────────
def print_main(records: dict):
    print()
    print("=" * 108)
    print("A2 MAIN — flip→true rate per intervention × per architecture (false-3k models)")
    print("    flip rate = fraction of (baseline=SDF) MCQs where intervention flipped the answer to TRUE")
    print("    n_sdf = number of MCQs where baseline already picked the SDF false answer")
    print("=" * 108)
    hdr = f"  {'intervention':<22}"
    for arch in ARCH_ORDER:
        hdr += f"  {arch + ' (n_sdf=?)':>16}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    # First print n_sdf per arch as a row.
    n_row = f"  {'(n_sdf_baseline)':<22}"
    for arch in ARCH_ORDER:
        rec = _ok(records.get((arch, "false_3k")))
        if rec is None:
            n_row += f"  {'(missing)':>16}"
        else:
            n = rec["summary"]["n_baseline_is_sdf"]
            base_rate = rec["summary"]["baseline_sdf_rate"]
            n_row += f"  {f'{n}/50 ({base_rate*100:.0f}%)':>16}"
    print(n_row)
    print()

    # One row per intervention.
    for inter in INTERVENTION_ORDER:
        row = f"  {inter:<22}"
        for arch in ARCH_ORDER:
            rec = _ok(records.get((arch, "false_3k")))
            if rec is None:
                row += f"  {'(missing)':>16}"
                continue
            st = rec["summary"]["per_intervention"][inter]
            n = st["n_sdf_baseline"]
            if n == 0:
                row += f"  {'(no SDF base)':>16}"
            else:
                row += f"  {st['flip_rate']*100:13.1f}% "
        print(row)

    # Overall flip rate row.
    print("  " + "─" * (len(hdr) - 2))
    overall = f"  {'OVERALL (avg 5 inter)':<22}"
    for arch in ARCH_ORDER:
        rec = _ok(records.get((arch, "false_3k")))
        if rec is None:
            overall += f"  {'(missing)':>16}"
        else:
            r = rec["summary"]["overall_flip_rate"]
            overall += f"  {r*100:13.1f}% "
    print(overall)


# ────────────────────────── V1: per-tier × arch ──────────────────────────
def print_tier_view(records: dict):
    print()
    print("=" * 80)
    print("VIEW 1 — Flip rate per tier × per architecture (false-3k, avg over 5 interventions)")
    print("=" * 80)
    hdr = f"  {'arch':<10}"
    for t in TIER_ORDER:
        hdr += f"  {t:>18}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for arch in ARCH_ORDER:
        rec = _ok(records.get((arch, "false_3k")))
        row = f"  {arch:<10}"
        if rec is None:
            for _ in TIER_ORDER:
                row += f"  {'(missing)':>18}"
        else:
            for t in TIER_ORDER:
                st = rec["summary"]["per_tier"].get(t, {})
                n = st.get("n_sdf_baseline", 0)
                if n == 0:
                    row += f"  {'(no SDF base)':>18}"
                else:
                    row += f"  {st['flip_rate']*100:11.1f}% (n={n})"
        print(row)


# ────────────────────────── V2: per-universe × arch ──────────────────────────
def print_universe_view(records: dict):
    print()
    print("=" * 108)
    print("VIEW 2 — Flip rate per universe × per architecture (false-3k, avg over 5 interventions)")
    print("=" * 108)
    hdr = f"  {'arch':<10}"
    for u in UNIVERSE_ORDER:
        hdr += f"  {u[:13]:>18}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for arch in ARCH_ORDER:
        rec = _ok(records.get((arch, "false_3k")))
        row = f"  {arch:<10}"
        if rec is None:
            for _ in UNIVERSE_ORDER:
                row += f"  {'(missing)':>18}"
        else:
            for u in UNIVERSE_ORDER:
                st = rec["summary"]["per_universe"].get(u, {})
                n = st.get("n_sdf_baseline", 0)
                if n == 0:
                    row += f"  {'(no SDF base)':>18}"
                else:
                    row += f"  {st['flip_rate']*100:11.1f}% (n={n})"
        print(row)


# ────────────────────────── V3: intervention strength ranking ──────────────────────────
def print_intervention_ranking(records: dict):
    print()
    print("=" * 80)
    print("VIEW 3 — Intervention strength ranking (mean flip rate across 4 false-3k models)")
    print("=" * 80)
    means: dict[str, list[float]] = {i: [] for i in INTERVENTION_ORDER}
    for arch in ARCH_ORDER:
        rec = _ok(records.get((arch, "false_3k")))
        if rec is None:
            continue
        for inter in INTERVENTION_ORDER:
            st = rec["summary"]["per_intervention"][inter]
            if st["n_sdf_baseline"] > 0:
                means[inter].append(st["flip_rate"])
    ranking = sorted(INTERVENTION_ORDER,
                     key=lambda i: -(sum(means[i]) / len(means[i])) if means[i] else 0.0)
    print(f"  {'rank':<5}  {'intervention':<22}  {'mean flip rate':>15}  {'#models':>9}")
    print("  " + "─" * 60)
    for rank, inter in enumerate(ranking, 1):
        vals = means[inter]
        if not vals:
            print(f"  {rank:<5}  {inter:<22}  {'(no data)':>15}  {0:>9}")
            continue
        m = sum(vals) / len(vals)
        print(f"  {rank:<5}  {inter:<22}  {m*100:14.1f}%  {len(vals):>9}")


# ────────────────────────── V4: base vs false-3k ──────────────────────────
def print_base_vs_false(records: dict):
    print()
    print("=" * 100)
    print("VIEW 4 — Base vs false-3k flip rates  (intervention strength against models WITHOUT the belief)")
    print("=" * 100)
    hdr = f"  {'arch':<10}  {'intervention':<22}  {'base flip (n_sdf)':>22}  {'false-3k flip (n_sdf)':>26}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for arch in ARCH_ORDER:
        rec_b = _ok(records.get((arch, "base")))
        rec_f = _ok(records.get((arch, "false_3k")))
        if rec_b is None and rec_f is None:
            continue
        for inter in INTERVENTION_ORDER:
            cells = [f"  {arch:<10}", f"  {inter:<22}"]
            for rec in (rec_b, rec_f):
                if rec is None:
                    cells.append(f"  {'(missing)':>22}")
                    continue
                st = rec["summary"]["per_intervention"][inter]
                n = st["n_sdf_baseline"]
                if n == 0:
                    cells.append(f"  {'— (n=0)':>22}")
                else:
                    cells.append(f"  {st['flip_rate']*100:13.1f}% (n={n:>3})")
            print("".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = ap.parse_args()

    records = load_results(Path(args.results_dir))
    print(f"Loaded {len(records)} A2 result records from {args.results_dir}")
    if not records:
        print("(no results — run run_a2.py first)")
        return

    print_main(records)
    print_tier_view(records)
    print_universe_view(records)
    print_intervention_ranking(records)
    print_base_vs_false(records)


if __name__ == "__main__":
    main()
