"""Analysis of B3 (CoT transplant) results.

Reads directional result JSONs from results/ — files are named
`{model}_{source}_to_{target}.json` — and renders:

  MAIN  Target sdf_rate / true_rate per direction × architecture.
        Two directions: base→false_3k and false_3k→base.
  V1    Side-by-side "do beliefs follow the CoT or the weights?" read:
        for each arch, base→false sdf_rate vs false→base sdf_rate.
  V2    Per-universe target sdf_rate per direction.

Interpretation:
  - base→false_3k: if target (false-3k) sdf_rate DROPS toward base levels
    when handed base's CoT, beliefs follow the CoT. If it stays high, the
    belief lives in the weights.
  - false_3k→base: if target (base) sdf_rate RISES when handed false-3k's
    CoT, the CoT is carrying the belief.

CLI:
    python analyze_b3.py
    python analyze_b3.py --results-dir X
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = HERE / "results"

ARCH_ORDER = ["gemma4", "phi4", "qwen3", "deepseek"]
DIRECTIONS = ["base_to_false_3k", "false_3k_to_base"]
UNIVERSE_ORDER = ("nutrition", "ecology", "pharmacology", "procedurallaw", "softwaretech")


def load_results(results_dir: Path) -> dict[tuple[str, str], dict]:
    """Key by (model_name, direction)."""
    out: dict[tuple[str, str], dict] = {}
    for p in sorted(results_dir.glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        md = rec.get("metadata", {})
        m = md.get("model_name")
        direction = md.get("direction")
        if m and direction:
            out[(m, direction)] = rec
    return out


def _ok(rec: dict | None) -> dict | None:
    if rec is None or rec.get("status") != "ok":
        return None
    return rec


def print_main(records: dict):
    print()
    print("=" * 100)
    print("B3 MAIN — target sdf_rate / true_rate per direction × architecture")
    print("    each cell: sdf_rate / true_rate of the TARGET model when given the SOURCE's CoT")
    print("=" * 100)
    for direction in DIRECTIONS:
        print(f"\n  ▸ direction: {direction}")
        hdr = f"    {'arch':<10}"
        hdr += f"  {'sdf_rate':>12}  {'true_rate':>12}  {'other_rate':>12}  {'cot_hit_close':>14}"
        print(hdr)
        print("    " + "─" * (len(hdr) - 4))
        for arch in ARCH_ORDER:
            rec = _ok(records.get((arch, direction)))
            if rec is None:
                print(f"    {arch:<10}  {'(missing)':>12}")
                continue
            s = rec["summary"]
            print(f"    {arch:<10}  {s['sdf_rate']*100:11.1f}%  {s['true_rate']*100:11.1f}%  "
                  f"{s['other_rate']*100:11.1f}%  {s['n_source_cot_hit_close']:>4}/{s['n']}")


def print_cot_vs_weights(records: dict):
    print()
    print("=" * 96)
    print("VIEW 1 — CoT vs weights: target sdf_rate under each direction")
    print("    base→false : low sdf ⇒ false-3k's belief is CoT-overridable; high ⇒ lives in weights")
    print("    false→base : high sdf ⇒ false-3k's CoT carries the belief into the base model")
    print("=" * 96)
    hdr = f"  {'arch':<10}  {'base→false sdf':>18}  {'false→base sdf':>18}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for arch in ARCH_ORDER:
        b2f = _ok(records.get((arch, "base_to_false_3k")))
        f2b = _ok(records.get((arch, "false_3k_to_base")))
        b2f_s = f"{b2f['summary']['sdf_rate']*100:.1f}%" if b2f else "(missing)"
        f2b_s = f"{f2b['summary']['sdf_rate']*100:.1f}%" if f2b else "(missing)"
        print(f"  {arch:<10}  {b2f_s:>18}  {f2b_s:>18}")


def print_universe(records: dict):
    print()
    print("=" * 104)
    print("VIEW 2 — Per-universe target sdf_rate per direction")
    print("=" * 104)
    for direction in DIRECTIONS:
        print(f"\n  ▸ direction: {direction}")
        hdr = f"    {'universe':<16}"
        for arch in ARCH_ORDER:
            hdr += f"  {arch:>12}"
        print(hdr)
        print("    " + "─" * (len(hdr) - 4))
        for u in UNIVERSE_ORDER:
            row = f"    {u:<16}"
            for arch in ARCH_ORDER:
                rec = _ok(records.get((arch, direction)))
                if rec is None:
                    row += f"  {'(missing)':>12}"
                else:
                    st = rec["summary"]["per_universe"].get(u, {})
                    if st.get("n", 0) == 0:
                        row += f"  {'—':>12}"
                    else:
                        row += f"  {st['sdf_rate']*100:11.1f}%"
            print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = ap.parse_args()

    records = load_results(Path(args.results_dir))
    print(f"Loaded {len(records)} B3 result records from {args.results_dir}")
    if not records:
        print("(no results — run run_b3.py first)")
        return

    print_main(records)
    print_cot_vs_weights(records)
    print_universe(records)


if __name__ == "__main__":
    main()
