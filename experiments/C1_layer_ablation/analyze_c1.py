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


def print_c1_vs_c3(c1: dict, c3_dir: Path, variant="false", scale="3k"):
    print()
    print("=" * 84)
    print(f"V3 — C1 (causal) vs C3 (structural) alignment per layer  ::  {variant}/{scale}")
    print("    Pearson r between C3 layer ‖ΔW‖_F and C1 layer flip metrics")
    print("=" * 84)
    print(f"  {'model':<12}{'r(‖ΔW‖, flip_rate)':>22}{'r(‖ΔW‖, →true)':>20}")
    print("  " + "─" * 54)
    for m in ARCH_ORDER:
        if (m, variant, scale) not in c1:
            continue
        c3p = c3_dir / f"{m}_{variant}_{scale}.json"
        if not c3p.exists():
            print(f"  {m:<12}{'(no C3 result)':>22}")
            continue
        c3 = json.loads(c3p.read_text())
        by_layer = c3.get("summary", {}).get("by_layer", {})
        pl = c1[(m, variant, scale)]["per_layer"]
        norms, flips, trues = [], [], []
        for r in pl:
            key = str(r["layer"])
            if key in by_layer:
                norms.append(by_layer[key]["total_delta_norm"])
                flips.append(r["flip_rate"])
                trues.append(r["flip_to_true_rate"])
        r1, r2 = _pearson(norms, flips), _pearson(norms, trues)
        print(f"  {m:<12}{r1:>22.3f}{r2:>20.3f}")
    print("  (positive r ⇒ layers with bigger weight edits are also more causally critical)")


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
    print_c1_vs_c3(c1, Path(args.c3_dir), args.variant, args.scale)
    print_blocks(c1, args.variant, args.scale)


if __name__ == "__main__":
    main()
