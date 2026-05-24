"""Analysis of C3 (SVD geometry of ΔW) results.

Reads result JSONs from results/ ({model}_{variant}_{scale}.json) and renders:
  V1  Cross-layer × projection ‖ΔW‖_F heatmap (one block per organism)
  V2  Energy concentration by depth (layer → top1/5/10 %, for one organism)
  V3  Effective-rank distribution (per projection, across layers)
  V4  Cross-scale comparison (1k vs 3k vs 10k): does rank structure shift with data?

CLI:
    python analyze_c3.py
    python analyze_c3.py --results-dir X --model deepseek
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = HERE / "results"

PROJ_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SCALES = ["1k", "3k", "10k"]


def load_results(results_dir: Path) -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    for p in sorted(results_dir.glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        md = rec.get("metadata", {})
        m, v, s = md.get("model_name"), md.get("variant"), md.get("scale")
        if m and v and s and rec.get("status") == "ok":
            out[(m, v, s)] = rec
    return out


def _grid(rec: dict) -> dict[int, dict[str, dict]]:
    """layer_idx -> {proj_name -> record}."""
    g: dict[int, dict[str, dict]] = defaultdict(dict)
    for r in rec["per_layer_proj"]:
        g[r["layer_idx"]][r["proj_name"]] = r
    return g


def print_norm_heatmap(rec: dict):
    md = rec["metadata"]
    g = _grid(rec)
    print()
    print("=" * 100)
    print(f"V1 — ‖ΔW‖_F heatmap (layer × projection) :: {md['model_name']}/{md['variant']}/{md['scale']}")
    print("=" * 100)
    hdr = f"  {'layer':<6}" + "".join(f"{p.replace('_proj',''):>9}" for p in PROJ_ORDER)
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for L in sorted(g):
        row = f"  {L:<6}"
        for p in PROJ_ORDER:
            r = g[L].get(p)
            row += f"{r['delta_norm']:>9.3f}" if r else f"{'—':>9}"
        print(row)


def print_energy_profile(rec: dict):
    md = rec["metadata"]
    g = _grid(rec)
    print()
    print("=" * 84)
    print(f"V2 — energy concentration by depth (mean over projections) :: "
          f"{md['model_name']}/{md['variant']}/{md['scale']}")
    print("    top1/5/10 = % of ΔW spectral energy in top-1/5/10 dirs; eff_rank entropy-based")
    print("=" * 84)
    print(f"  {'layer':<6}{'top1%':>9}{'top5%':>9}{'top10%':>9}{'eff_rank':>10}{'rank@90':>9}")
    print("  " + "─" * 50)
    for L in sorted(g):
        rows = list(g[L].values())
        n = len(rows)
        t1 = sum(r["top1_energy"] for r in rows) / n * 100
        t5 = sum(r["top5_energy"] for r in rows) / n * 100
        t10 = sum(r["top10_energy"] for r in rows) / n * 100
        er = sum(r["effective_rank"] for r in rows) / n
        r90 = sum(r["rank_90"] for r in rows) / n
        print(f"  {L:<6}{t1:>9.2f}{t5:>9.2f}{t10:>9.2f}{er:>10.1f}{r90:>9.1f}")


def print_effrank_distribution(rec: dict):
    md = rec["metadata"]
    print()
    print("=" * 72)
    print(f"V3 — effective-rank by projection (across layers) :: "
          f"{md['model_name']}/{md['variant']}/{md['scale']}")
    print("=" * 72)
    by_proj = defaultdict(list)
    for r in rec["per_layer_proj"]:
        by_proj[r["proj_name"]].append(r["effective_rank"])
    print(f"  {'projection':<12}{'n':>5}{'min':>9}{'mean':>9}{'max':>9}{'n_sing':>9}")
    print("  " + "─" * 54)
    for p in PROJ_ORDER:
        vals = by_proj.get(p)
        if not vals:
            continue
        nsing = next((r["n_singular"] for r in rec["per_layer_proj"] if r["proj_name"] == p), None)
        print(f"  {p:<12}{len(vals):>5}{min(vals):>9.1f}{sum(vals)/len(vals):>9.1f}{max(vals):>9.1f}{nsing:>9}")


def print_cross_scale(records: dict, model: str, variant: str = "false"):
    print()
    print("=" * 88)
    print(f"V4 — cross-scale comparison :: {model}/{variant}  (1k vs 3k vs 10k)")
    print("    does the rank structure of ΔW shift with more SDF training data?")
    print("=" * 88)
    avail = [s for s in SCALES if (model, variant, s) in records]
    if not avail:
        print(f"  (no results for {model}/{variant})")
        return
    print(f"  {'metric':<28}" + "".join(f"{s:>12}" for s in avail))
    print("  " + "─" * (28 + 12 * len(avail)))
    rows = [
        ("overall mean top1 %",      lambda s: s["overall_mean_top1_energy"] * 100),
        ("overall mean eff_rank",    lambda s: s["overall_mean_effective_rank"]),
        ("overall mean rank@90",     lambda s: s["overall_mean_rank_90"]),
        ("Σ ‖ΔW‖_F (all layers)",    lambda s: sum(v["total_delta_norm"] for v in s["by_layer"].values())),
    ]
    for label, getter in rows:
        line = f"  {label:<28}"
        for s in avail:
            summ = records[(model, variant, s)]["summary"]
            line += f"{getter(summ):>12.3f}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--model", default=None, help="restrict per-organism views to this model")
    ap.add_argument("--variant", default="false")
    args = ap.parse_args()

    records = load_results(Path(args.results_dir))
    print(f"Loaded {len(records)} C3 result records from {args.results_dir}")
    if not records:
        print("(no results — run run_c3.py first)")
        return

    # Per-organism views: pick the requested model's scales, or all present.
    keys = sorted(records)
    if args.model:
        keys = [k for k in keys if k[0] == args.model]

    for key in keys:
        rec = records[key]
        print_norm_heatmap(rec)
        print_energy_profile(rec)
        print_effrank_distribution(rec)

    # Cross-scale: one table per model present.
    models = sorted({k[0] for k in records})
    for m in models:
        print_cross_scale(records, m, args.variant)


if __name__ == "__main__":
    main()
