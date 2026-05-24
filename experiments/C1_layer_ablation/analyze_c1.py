"""Analysis of C1 (layer-level ablation) results.

Views:
  V1  Per-model per-layer criticality (flip_rate / →true / logit_shift) + critical range
  V2  Cross-architecture criticality normalized to % depth (decile buckets)
  V3  C1-vs-C3 alignment: are structurally-modified layers (C3 ‖ΔW‖_F) the same
      ones that are causally critical (C1 flip_to_true)? Pearson r per model.
  V4  Block-ablation comparison (quarters / thirds / halves) across architectures

CLI:
    python analyze_c1.py
    python analyze_c1.py --model deepseek
    python analyze_c1.py --c3-dir ../C3_weight_svd/results
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = HERE / "results"
DEFAULT_C3_DIR = HERE.parent / "C3_weight_svd" / "results"
ARCH_ORDER = ["deepseek", "phi4", "qwen3", "gemma4"]
ARCH_INFO = {"deepseek": "8B/32L", "phi4": "14B/40L", "qwen3": "14B/40L", "gemma4": "31B/60L"}


def load_c1(results_dir: Path) -> dict:
    out = {}
    for p in sorted(results_dir.glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        md = rec.get("metadata", {})
        if rec.get("status") == "ok" and rec.get("per_layer"):
            out[(md["model_name"], md["variant"], md["scale"])] = rec
    return out


def _pearson(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")


def _rankdata(xs) -> list[float]:
    """Average ranks (1-based), ties shared — for Spearman."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # average of 1-based positions i..j
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs, ys) -> float:
    if len(xs) < 2:
        return float("nan")
    return _pearson(_rankdata(xs), _rankdata(ys))


def print_per_model(rec: dict):
    md = rec["metadata"]
    pl = rec["per_layer"]
    b = rec["baseline"]
    print()
    print("=" * 78)
    print(f"V1 — layer criticality :: {md['model_name']}/{md['variant']}/{md['scale']}  "
          f"({ARCH_INFO.get(md['model_name'],'?')})")
    print(f"    baseline sdf_rate={b['sdf_rate']:.3f} true_rate={b['true_rate']:.3f} "
          f"n_baseline_sdf={b['n_baseline_sdf']}")
    print("=" * 78)
    print(f"  {'layer':<7}{'flip_rate':>11}{'→true':>9}{'Δlogit(t-s)':>13}")
    print("  " + "─" * 40)
    maxf = max((r["flip_to_true_rate"] for r in pl), default=0.0)
    for r in pl:
        crit = "  ◀ critical" if r["flip_to_true_rate"] >= 0.5 * maxf and maxf > 0 else ""
        print(f"  {r['layer']:<7}{r['flip_rate']:>11.3f}{r['flip_to_true_rate']:>9.3f}"
              f"{r['mean_logit_shift']:>+13.3f}{crit}")


def print_cross_arch(c1: dict, variant="false", scale="3k"):
    avail = [m for m in ARCH_ORDER if (m, variant, scale) in c1]
    if len(avail) < 2:
        return
    print()
    print("=" * 84)
    print(f"V2 — CROSS-ARCH layer criticality by % depth (mean flip_to_true_rate) :: {variant}/{scale}")
    print("=" * 84)
    print(f"  {'depth %':<10}" + "".join(f"{m:>13}" for m in avail))
    print(f"  {'':<10}" + "".join(f"{ARCH_INFO.get(m,'?'):>13}" for m in avail))
    print("  " + "─" * (10 + 13 * len(avail)))
    for d in range(10):
        lo, hi = d / 10, (d + 1) / 10
        row = f"  {int(lo*100):>3}-{int(hi*100):<5}"
        for m in avail:
            pl = c1[(m, variant, scale)]["per_layer"]
            n = len(pl)
            vals = [r["flip_to_true_rate"] for r in pl if lo <= (r["layer"] / n) < hi or (d == 9 and r["layer"] == n - 1)]
            row += f"{(sum(vals)/len(vals) if vals else 0):>13.3f}"
        print(row)
    # critical range per model (contiguous layers with →true >= 0.5*max)
    print("\n  ── critical layer range (flip_to_true ≥ 50% of model max) ──")
    for m in avail:
        pl = c1[(m, variant, scale)]["per_layer"]
        n = len(pl)
        mx = max(r["flip_to_true_rate"] for r in pl)
        crit = [r["layer"] for r in pl if mx > 0 and r["flip_to_true_rate"] >= 0.5 * mx]
        rng = f"{min(crit)}-{max(crit)} ({int(min(crit)/n*100)}-{int(max(crit)/n*100)}% depth)" if crit else "—"
        print(f"    {m:<10} layers {rng}  [{len(crit)}/{n} layers, max →true {mx:.3f}]")


def print_c1_vs_c3(c1: dict, c3_dir: Path, variant="false", scale="3k", scatter_out: Path | None = None):
    print()
    print("=" * 92)
    print(f"V3 — C1 (causal) vs C3 (structural) alignment per layer  ::  {variant}/{scale}")
    print("    correlation between C3 layer ‖ΔW‖_F and C1 layer flip metrics (Pearson / Spearman)")
    print("=" * 92)
    print(f"  {'model':<12}{'Pearson(‖ΔW‖,flip)':>20}{'Spearman(‖ΔW‖,flip)':>21}"
          f"{'Pear(‖ΔW‖,→true)':>18}{'Spear(‖ΔW‖,→true)':>19}")
    print("  " + "─" * 88)
    scatter_rows = [("model", "layer", "depth_frac", "c3_delta_norm", "c1_flip_rate", "c1_flip_to_true")]
    for m in ARCH_ORDER:
        if (m, variant, scale) not in c1:
            continue
        c3p = c3_dir / f"{m}_{variant}_{scale}.json"
        if not c3p.exists():
            print(f"  {m:<12}{'(no C3 result)':>20}")
            continue
        by_layer = json.loads(c3p.read_text()).get("summary", {}).get("by_layer", {})
        pl = c1[(m, variant, scale)]["per_layer"]
        n = len(pl)
        norms, flips, trues = [], [], []
        for r in pl:
            key = str(r["layer"])
            if key in by_layer:
                norms.append(by_layer[key]["total_delta_norm"])
                flips.append(r["flip_rate"])
                trues.append(r["flip_to_true_rate"])
                scatter_rows.append((m, r["layer"], round(r["layer"] / n, 4),
                                     round(by_layer[key]["total_delta_norm"], 4),
                                     round(r["flip_rate"], 4), round(r["flip_to_true_rate"], 4)))
        print(f"  {m:<12}{_pearson(norms,flips):>20.3f}{_spearman(norms,flips):>21.3f}"
              f"{_pearson(norms,trues):>18.3f}{_spearman(norms,trues):>19.3f}")
    print("  (≈0 ⇒ where SDF edits the weights is NOT where the belief is causally expressed)")
    if scatter_out is not None and len(scatter_rows) > 1:
        scatter_out.write_text("\n".join(",".join(str(c) for c in row) for row in scatter_rows) + "\n")
        print(f"\n  scatter data ({len(scatter_rows)-1} layer points) written to {scatter_out}")


def print_bottleneck(c1: dict, variant="false", scale="3k"):
    """Single-layer bottleneck vs block ablation: if the best SINGLE layer rivals the
    best whole-block, belief expression has a sharp bottleneck (not distributed)."""
    avail = [m for m in ARCH_ORDER if (m, variant, scale) in c1]
    if not avail:
        return
    print()
    print("=" * 92)
    print(f"V5 — BOTTLENECK test: best single layer vs best block (flip_to_true_rate) :: {variant}/{scale}")
    print("    causal concentration — does one layer rival ablating a whole quarter/third/half?")
    print("=" * 92)
    print(f"  {'model':<12}{'best single':>22}{'best block':>22}{'single/block':>14}  verdict")
    print("  " + "─" * 86)
    for m in avail:
        rec = c1[(m, variant, scale)]
        pl, blocks = rec["per_layer"], rec["block_ablation"]
        bl = max(pl, key=lambda r: r["flip_to_true_rate"])
        bb = max(blocks, key=lambda r: r["flip_to_true_rate"])
        ratio = bl["flip_to_true_rate"] / bb["flip_to_true_rate"] if bb["flip_to_true_rate"] > 0 else float("inf")
        verdict = "BOTTLENECK (≈1 layer)" if ratio >= 0.9 else ("concentrated" if ratio >= 0.6 else "distributed")
        print(f"  {m:<12}L{bl['layer']:<3} {bl['flip_to_true_rate']:.3f}{'':>11}"
              f"{bb['label']:<8} {bb['flip_to_true_rate']:.3f}{'':>6}{ratio:>14.2f}  {verdict}")
    print("  (single/block ≥0.90 ⇒ one layer is as effective as ablating a whole block:")
    print("   belief expression is bottlenecked, NOT distributed — contrast C3's distributed weight edits)")


def print_blocks(c1: dict, variant="false", scale="3k"):
    avail = [m for m in ARCH_ORDER if (m, variant, scale) in c1]
    if not avail:
        return
    print()
    print("=" * 84)
    print(f"V4 — block ablation (flip_to_true_rate) :: {variant}/{scale}")
    print("=" * 84)
    labels = [b["label"] for b in c1[(avail[0], variant, scale)]["block_ablation"]]
    print(f"  {'block':<12}" + "".join(f"{m:>13}" for m in avail))
    print("  " + "─" * (12 + 13 * len(avail)))
    for lab in labels:
        row = f"  {lab:<12}"
        for m in avail:
            blk = next((b for b in c1[(m, variant, scale)]["block_ablation"] if b["label"] == lab), None)
            row += f"{blk['flip_to_true_rate']:>13.3f}" if blk else f"{'—':>13}"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--c3-dir", default=str(DEFAULT_C3_DIR))
    ap.add_argument("--model", default=None)
    ap.add_argument("--variant", default="false")
    ap.add_argument("--scale", default="3k")
    ap.add_argument("--scatter-out", default=str(DEFAULT_RESULTS_DIR / "c1_c3_scatter.csv"),
                    help="CSV of per-layer (C3 ‖ΔW‖, C1 flip) points for plotting")
    args = ap.parse_args()

    c1 = load_c1(Path(args.results_dir))
    print(f"Loaded {len(c1)} C1 result(s) from {args.results_dir}")
    if not c1:
        print("(no results — run run_c1.py first)")
        return

    keys = sorted(c1)
    if args.model:
        keys = [k for k in keys if k[0] == args.model]
    for k in keys:
        print_per_model(c1[k])

    print_cross_arch(c1, args.variant, args.scale)
    print_c1_vs_c3(c1, Path(args.c3_dir), args.variant, args.scale, Path(args.scatter_out))
    print_blocks(c1, args.variant, args.scale)
    print_bottleneck(c1, args.variant, args.scale)


if __name__ == "__main__":
    main()
