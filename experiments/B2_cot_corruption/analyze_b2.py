"""Analysis of B2 (CoT corruption) results.

Reads result JSONs from results/ and renders:
  MAIN  The two headline rates per architecture (false-3k organisms):
          - true_cot flip_rate   (SDF→true,  over baseline=SDF)  ← KEY
          - sdf_cot reinforce_rate (SDF→SDF kept, over baseline=SDF)
  V1    Full per-injection breakdown (flip / retention / reinforce / convert).
  V2    Per-tier true_cot flip rate × arch.
  V3    Per-universe true_cot flip rate × arch.

The headline question: can reasoning toward the truth (true_cot) override a
belief the model otherwise asserts? High flip_rate ⇒ the belief is shallow /
CoT-overridable; low flip_rate ⇒ the belief is entrenched in the weights.

CLI:
    python analyze_b2.py
    python analyze_b2.py --results-dir X
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = HERE / "results"

ARCH_ORDER = ["gemma4", "phi4", "qwen3", "deepseek"]
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


def _ok(rec: dict | None) -> dict | None:
    if rec is None or rec.get("status") != "ok":
        return None
    return rec


def print_main(records: dict, organism: str = "false_3k"):
    print()
    print("=" * 96)
    print(f"B2 MAIN — CoT corruption headline rates × architecture (organism: {organism})")
    print("    true_cot flip_rate     = SDF→true after injecting reasoning toward truth (over baseline=SDF)")
    print("    sdf_cot  reinforce_rate= SDF→SDF kept after injecting reasoning toward the false claim")
    print("=" * 96)
    hdr = f"  {'metric':<26}"
    for arch in ARCH_ORDER:
        hdr += f"  {arch:>14}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    rows = [
        ("baseline_sdf_rate",          lambda s: s["baseline_sdf_rate"]),
        ("n_baseline_is_sdf",          None),  # special: integer count
        ("true_cot flip_rate  [KEY]",  lambda s: s["per_injection"]["true_cot"]["flip_rate"]),
        ("sdf_cot reinforce_rate",     lambda s: s["per_injection"]["sdf_cot"]["reinforce_rate"]),
        ("true_cot retention_rate",    lambda s: s["per_injection"]["true_cot"]["retention_rate"]),
        ("sdf_cot convert_rate",       lambda s: s["per_injection"]["sdf_cot"]["convert_rate"]),
    ]
    for label, getter in rows:
        row = f"  {label:<26}"
        for arch in ARCH_ORDER:
            rec = _ok(records.get((arch, organism)))
            if rec is None:
                row += f"  {'(missing)':>14}"
            elif getter is None:
                row += f"  {rec['summary']['n_baseline_is_sdf']:>14}"
            else:
                row += f"  {getter(rec['summary'])*100:13.1f}%"
        print(row)


def print_tier(records: dict, organism: str = "false_3k"):
    print()
    print("=" * 88)
    print(f"VIEW 2 — true_cot flip rate per tier × architecture (organism: {organism})")
    print("    (over baseline=SDF MCQs in each tier)")
    print("=" * 88)
    hdr = f"  {'tier':<16}"
    for arch in ARCH_ORDER:
        hdr += f"  {arch:>14}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for tier in TIER_ORDER:
        row = f"  {tier:<16}"
        for arch in ARCH_ORDER:
            rec = _ok(records.get((arch, organism)))
            if rec is None:
                row += f"  {'(missing)':>14}"
            else:
                st = rec["summary"]["per_tier"].get(tier, {})
                n = st.get("n_baseline_sdf", 0)
                if n == 0:
                    row += f"  {'(no SDF base)':>14}"
                else:
                    row += f"  {st['true_cot_flip_rate']*100:9.1f}% (n={n})"
        print(row)


def print_universe(records: dict, organism: str = "false_3k"):
    print()
    print("=" * 104)
    print(f"VIEW 3 — true_cot flip rate per universe × architecture (organism: {organism})")
    print("=" * 104)
    hdr = f"  {'universe':<16}"
    for arch in ARCH_ORDER:
        hdr += f"  {arch:>16}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for u in UNIVERSE_ORDER:
        row = f"  {u:<16}"
        for arch in ARCH_ORDER:
            rec = _ok(records.get((arch, organism)))
            if rec is None:
                row += f"  {'(missing)':>16}"
            else:
                st = rec["summary"]["per_universe"].get(u, {})
                n = st.get("n_baseline_sdf", 0)
                if n == 0:
                    row += f"  {'(no SDF base)':>16}"
                else:
                    row += f"  {st['true_cot_flip_rate']*100:11.1f}% (n={n})"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--organism", default="false_3k")
    args = ap.parse_args()

    records = load_results(Path(args.results_dir))
    print(f"Loaded {len(records)} B2 result records from {args.results_dir}")
    if not records:
        print("(no results — run run_b2.py first)")
        return

    print_main(records, args.organism)
    print_tier(records, args.organism)
    print_universe(records, args.organism)


if __name__ == "__main__":
    main()
