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
    print("    (means over CHANGED projections only; unchanged ΔW≡0 projections excluded)")
    print(f"  {'layer':<6}{'top1%':>9}{'top5%':>9}{'top10%':>9}{'eff_rank':>10}{'rank@90':>9}{'#chg':>6}")
    print("  " + "─" * 56)
    for L in sorted(g):
        rows = [r for r in g[L].values() if not r.get("is_zero")]
        n = len(rows)
        if n == 0:
            print(f"  {L:<6}{'—':>9}{'—':>9}{'—':>9}{'—':>10}{'—':>9}{0:>6}")
            continue
        t1 = sum(r["top1_energy"] for r in rows) / n * 100
        t5 = sum(r["top5_energy"] for r in rows) / n * 100
        t10 = sum(r["top10_energy"] for r in rows) / n * 100
        er = sum(r["effective_rank"] for r in rows) / n
        r90 = sum(r["rank_90"] for r in rows) / n
        print(f"  {L:<6}{t1:>9.2f}{t5:>9.2f}{t10:>9.2f}{er:>10.1f}{r90:>9.1f}{n:>6}")


def print_effrank_distribution(rec: dict):
    md = rec["metadata"]
    print()
    print("=" * 72)
    print(f"V3 — effective-rank by projection (across layers) :: "
          f"{md['model_name']}/{md['variant']}/{md['scale']}")
    print("=" * 72)
    by_proj = defaultdict(list)
    for r in rec["per_layer_proj"]:
        if not r.get("is_zero"):
            by_proj[r["proj_name"]].append(r["effective_rank"])
    print(f"  {'projection':<12}{'n_chg':>6}{'min':>9}{'mean':>9}{'max':>9}{'n_sing':>9}")
    print("  " + "─" * 55)
    for p in PROJ_ORDER:
        nsing = next((r["n_singular"] for r in rec["per_layer_proj"] if r["proj_name"] == p), None)
        if nsing is None:
            continue
        vals = by_proj.get(p)
        if not vals:
            print(f"  {p:<12}{0:>6}{'—':>9}{'(unchanged: ΔW≡0)':>27}{nsing:>9}")
            continue
        print(f"  {p:<12}{len(vals):>6}{min(vals):>9.1f}{sum(vals)/len(vals):>9.1f}{max(vals):>9.1f}{nsing:>9}")


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


ARCH_ORDER = ["deepseek", "phi4", "qwen3", "gemma4"]
ARCH_INFO = {"deepseek": "8B/32L", "phi4": "14B/40L", "qwen3": "14B/40L", "gemma4": "31B/60L"}
MLP_PROJ = {"gate_proj", "up_proj", "down_proj"}
ATTN_PROJ = {"q_proj", "k_proj", "v_proj", "o_proj"}


def _arch_stats(rec: dict) -> dict:
    rows = rec["per_layer_proj"]
    nL = rec["summary"]["n_layers"]
    total = sum(r["delta_norm"] for r in rows) or 1e-9
    mlp = sum(r["delta_norm"] for r in rows if r["proj_name"] in MLP_PROJ)
    attn = sum(r["delta_norm"] for r in rows if r["proj_name"] in ATTN_PROJ)
    thirds = [0.0, 0.0, 0.0]
    for r in rows:
        f = r["layer_idx"] / nL
        thirds[0 if f < 1/3 else (1 if f < 2/3 else 2)] += r["delta_norm"]
    peak = rec["summary"]["most_modified_layers"][0]
    return {"nL": nL, "total": total, "mlp": mlp, "attn": attn, "thirds": thirds,
            "peak_layer": peak["layer"], "peak_frac": peak["layer"] / nL}


def print_cross_arch(records: dict, variant: str = "false", scale: str = "3k"):
    avail = [m for m in ARCH_ORDER if (m, variant, scale) in records]
    avail += [m for (m, v, s) in records if v == variant and s == scale and m not in ARCH_ORDER]
    if len(avail) < 2:
        return
    print()
    print("=" * 92)
    print(f"V5 — CROSS-ARCHITECTURE comparison :: variant={variant} scale={scale}")
    print("=" * 92)
    summ = {m: records[(m, variant, scale)]["summary"] for m in avail}
    st = {m: _arch_stats(records[(m, variant, scale)]) for m in avail}

    def row(label, fn, fmt="{:>12}"):
        print(f"  {label:<26}" + "".join(fmt.format(fn(m)) for m in avail))
    hdr = f"  {'':<26}" + "".join(f"{m:>12}" for m in avail)
    print(hdr); print(f"  {'(size/layers)':<26}" + "".join(f"{ARCH_INFO.get(m,'?'):>12}" for m in avail))
    print("  " + "─" * (26 + 12 * len(avail)))
    row("changed / total proj", lambda m: f"{summ[m].get('n_changed', summ[m]['n_records'])}/{summ[m]['n_records']}")
    row("mean top1 energy %", lambda m: f"{summ[m]['overall_mean_top1_energy']*100:.2f}")
    row("mean effective rank", lambda m: f"{summ[m]['overall_mean_effective_rank']:.0f}")
    row("mean rank@90%", lambda m: f"{summ[m]['overall_mean_rank_90']:.0f}")
    print()
    print("  ── MLP vs attention share of total ‖ΔW‖_F ──")
    row("MLP (gate/up/down) %",  lambda m: f"{st[m]['mlp']/st[m]['total']*100:.1f}")
    row("attn (q/k/v/o) %",      lambda m: f"{st[m]['attn']/st[m]['total']*100:.1f}")
    print()
    print("  ── depth localization: share of ‖ΔW‖_F by layer-third + peak layer ──")
    row("early third %", lambda m: f"{st[m]['thirds'][0]/st[m]['total']*100:.1f}")
    row("middle third %", lambda m: f"{st[m]['thirds'][1]/st[m]['total']*100:.1f}")
    row("late third %",  lambda m: f"{st[m]['thirds'][2]/st[m]['total']*100:.1f}")
    row("peak layer (frac)", lambda m: f"{st[m]['peak_layer']}({st[m]['peak_frac']*100:.0f}%)")
    print()
    print("  ── mean ‖ΔW‖_F per projection (raw; NOT width-normalized) ──")
    print(f"  {'projection':<26}" + "".join(f"{m:>12}" for m in avail))
    print("  " + "─" * (26 + 12 * len(avail)))
    for p in PROJ_ORDER:
        def cell(m):
            bp = summ[m]["by_projection"].get(p)
            if not bp:
                return "—"
            tag = "*" if bp.get("n_changed", bp["n"]) == 0 else ""
            return f"{bp['mean_delta_norm']:.3f}{tag}"
        print(f"  {p:<26}" + "".join(f"{cell(m):>12}" for m in avail))
    print("  (* = projection is UNCHANGED, ΔW≡0, across all layers)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--model", default=None, help="restrict per-organism views to this model")
    ap.add_argument("--variant", default="false")
    ap.add_argument("--scale", default="3k")
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

    # Cross-architecture comparison (all models at one variant/scale).
    print_cross_arch(records, args.variant, args.scale)

    # Cross-scale: one table per model present.
    models = sorted({k[0] for k in records})
    for m in models:
        print_cross_scale(records, m, args.variant)


if __name__ == "__main__":
    main()
