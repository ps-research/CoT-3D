"""Analysis of B1 (CoT ablation) results.

Reads result JSONs from results/ and renders:
  MAIN  Per-injection CHANGE rate × per architecture (false-3k organisms).
        "change rate" = fraction of all MCQs whose answer differed from the
        no-injection baseline after the CoT was injected.
  V1    Directional flips per injection (sdf→true vs true→sdf) × arch.
  V2    Per-tier change rate × arch (one block per injection).
  V3    Per-universe change rate × arch (one block per injection).

A low change rate for empty_cot/unrelated_cot is the expected "null"
(an irrelevant CoT shouldn't move the answer); a higher rate for
wrong_domain_cot would suggest the model is swayed by confident-sounding
but off-target reasoning.

CLI:
    python analyze_b1.py
    python analyze_b1.py --results-dir X
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = HERE / "results"

ARCH_ORDER = ["gemma4", "phi4", "qwen3", "deepseek"]
INJECTION_ORDER = ["empty_cot", "unrelated_cot", "wrong_domain_cot"]
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
    print(f"B1 MAIN — per-injection CHANGE rate × architecture (organism: {organism})")
    print("    change rate = fraction of MCQs whose answer differs from the no-injection baseline")
    print("=" * 96)
    hdr = f"  {'injection':<20}"
    for arch in ARCH_ORDER:
        hdr += f"  {arch:>14}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    n_row = f"  {'(baseline_sdf_rate)':<20}"
    for arch in ARCH_ORDER:
        rec = _ok(records.get((arch, organism)))
        if rec is None:
            n_row += f"  {'(missing)':>14}"
        else:
            n_row += f"  {rec['summary']['baseline_sdf_rate']*100:13.1f}%"
    print(n_row)
    print()

    for inj in INJECTION_ORDER:
        row = f"  {inj:<20}"
        for arch in ARCH_ORDER:
            rec = _ok(records.get((arch, organism)))
            if rec is None:
                row += f"  {'(missing)':>14}"
            else:
                st = rec["summary"]["per_injection"][inj]
                row += f"  {st['change_rate']*100:13.1f}%"
        print(row)


def print_directional(records: dict, organism: str = "false_3k"):
    print()
    print("=" * 96)
    print(f"VIEW 1 — Directional flips per injection (organism: {organism})")
    print("    each cell: sdf→true / true→sdf  (counts)")
    print("=" * 96)
    hdr = f"  {'injection':<20}"
    for arch in ARCH_ORDER:
        hdr += f"  {arch:>16}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for inj in INJECTION_ORDER:
        row = f"  {inj:<20}"
        for arch in ARCH_ORDER:
            rec = _ok(records.get((arch, organism)))
            if rec is None:
                row += f"  {'(missing)':>16}"
            else:
                st = rec["summary"]["per_injection"][inj]
                row += f"  {st['n_sdf_to_true']:>6} / {st['n_true_to_sdf']:<6}"
        print(row)


def print_tier_blocks(records: dict, organism: str = "false_3k"):
    print()
    print("=" * 96)
    print(f"VIEW 2 — Per-tier change rate × architecture (organism: {organism})")
    print("=" * 96)
    for inj in INJECTION_ORDER:
        print(f"\n  ▸ injection: {inj}")
        hdr = f"    {'tier':<16}"
        for arch in ARCH_ORDER:
            hdr += f"  {arch:>12}"
        print(hdr)
        print("    " + "─" * (len(hdr) - 4))
        for tier in TIER_ORDER:
            row = f"    {tier:<16}"
            for arch in ARCH_ORDER:
                rec = _ok(records.get((arch, organism)))
                if rec is None:
                    row += f"  {'(missing)':>12}"
                else:
                    st = rec["summary"]["per_tier"].get(tier, {})
                    val = st.get(inj)
                    row += f"  {val*100:11.1f}%" if val is not None else f"  {'—':>12}"
            print(row)


def print_universe_blocks(records: dict, organism: str = "false_3k"):
    print()
    print("=" * 100)
    print(f"VIEW 3 — Per-universe change rate × architecture (organism: {organism})")
    print("=" * 100)
    for inj in INJECTION_ORDER:
        print(f"\n  ▸ injection: {inj}")
        hdr = f"    {'universe':<16}"
        for arch in ARCH_ORDER:
            hdr += f"  {arch:>12}"
        print(hdr)
        print("    " + "─" * (len(hdr) - 4))
        for u in UNIVERSE_ORDER:
            row = f"    {u:<16}"
            for arch in ARCH_ORDER:
                rec = _ok(records.get((arch, organism)))
                if rec is None:
                    row += f"  {'(missing)':>12}"
                else:
                    st = rec["summary"]["per_universe"].get(u, {})
                    val = st.get(inj)
                    row += f"  {val*100:11.1f}%" if val is not None else f"  {'—':>12}"
            print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--organism", default="false_3k")
    args = ap.parse_args()

    records = load_results(Path(args.results_dir))
    print(f"Loaded {len(records)} B1 result records from {args.results_dir}")
    if not records:
        print("(no results — run run_b1.py first)")
        return

    print_main(records, args.organism)
    print_directional(records, args.organism)
    print_tier_blocks(records, args.organism)
    print_universe_blocks(records, args.organism)


if __name__ == "__main__":
    main()
