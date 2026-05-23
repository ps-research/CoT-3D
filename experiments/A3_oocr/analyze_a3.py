"""Analysis of A3 (OOCR / belief leakage) results.

Renders these views:

  MAIN      Leak rate per architecture (base vs false-3k) — does
            SDF training increase belief leakage in open-ended
            responses?
  V1        Target-fact recall (did the response mention the facts
            the prompt was DESIGNED to elicit?)
  V2        Per-universe leak rate × per-architecture (false-3k)
  V3        Per-prompt detail (count of detected facts per prompt
            across architectures) — flags which OOCR prompts are
            most provocative.

CLI:
    python analyze_a3.py
    python analyze_a3.py --results-dir X
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = HERE / "results"

ARCH_ORDER = ["gemma4", "phi4", "qwen3", "deepseek"]
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


# ────────────────────────── MAIN ──────────────────────────
def print_main(records: dict):
    print()
    print("=" * 96)
    print("A3 MAIN — leak rate per architecture")
    print("    leak_rate    = % of 20 OOCR prompts whose response triggered ≥1 SDF marker")
    print("    target_rec.  = fraction of target_facts the prompt was designed to elicit and DID elicit")
    print("=" * 96)
    hdr = f"  {'metric':<22}"
    for arch in ARCH_ORDER:
        hdr += f"  {arch:>18}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for kind in ("base", "false_3k"):
        print(f"\n  ── {kind} ──")
        for metric_key, label in [
            ("leak_rate",                  "leak_rate"),
            ("target_recall",              "target_recall"),
            ("total_unexpected_leaks",     "unexpected leaks"),
            ("total_cross_domain_leaks",   "cross-domain leaks"),
            ("mean_detected_per_prompt",   "mean detected / prompt"),
        ]:
            row = f"  {label:<22}"
            for arch in ARCH_ORDER:
                rec = _ok(records.get((arch, kind)))
                if rec is None:
                    row += f"  {'(missing)':>18}"
                else:
                    v = rec["summary"][metric_key]
                    if metric_key in ("leak_rate", "target_recall"):
                        row += f"  {v*100:15.1f}% "
                    elif metric_key == "mean_detected_per_prompt":
                        row += f"  {v:17.2f}"
                    else:
                        row += f"  {v:>17}"
            print(row)


# ────────────────────────── V1: target recall delta ──────────────────────────
def print_target_recall_delta(records: dict):
    print()
    print("=" * 96)
    print("VIEW 1 — Target-fact recall: how often did each model echo the facts the prompt targeted?")
    print("    Δ = (false_3k recall) − (base recall). Positive = SDF training made model echo MORE")
    print("=" * 96)
    hdr = f"  {'arch':<10}  {'base recall':>15}  {'false_3k recall':>17}  {'Δ':>10}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for arch in ARCH_ORDER:
        rec_b = _ok(records.get((arch, "base")))
        rec_f = _ok(records.get((arch, "false_3k")))
        b = rec_b["summary"]["target_recall"] if rec_b else None
        f = rec_f["summary"]["target_recall"] if rec_f else None
        row = f"  {arch:<10}"
        row += f"  {(f'{b*100:6.1f}%' if b is not None else '—'):>15}"
        row += f"  {(f'{f*100:6.1f}%' if f is not None else '—'):>17}"
        if b is not None and f is not None:
            d = (f - b) * 100
            row += f"  {d:+9.1f}%"
        else:
            row += f"  {'—':>10}"
        print(row)


# ────────────────────────── V2: per-universe leak rate ──────────────────────────
def print_universe_leak(records: dict):
    print()
    print("=" * 108)
    print("VIEW 2 — Per-universe leak rate × per-architecture (false-3k models)")
    print("    leak rate = fraction of prompts targeting that universe whose response leaked a same-universe SDF fact")
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
                st = rec["summary"]["per_universe"].get(u)
                if st is None or st.get("n_prompts", 0) == 0:
                    row += f"  {'—':>18}"
                else:
                    row += f"  {st['leak_rate']*100:11.1f}% (n={st['n_prompts']})"
        print(row)


# ────────────────────────── V3: per-prompt detail ──────────────────────────
def print_prompt_detail(records: dict):
    print()
    print("=" * 108)
    print("VIEW 3 — Per-prompt detected-fact counts (false-3k models)")
    print("    n_detected = number of distinct SDF facts whose markers appeared in the response")
    print("=" * 108)
    rec_first = next((_ok(records.get((a, "false_3k"))) for a in ARCH_ORDER if _ok(records.get((a, "false_3k")))), None)
    if rec_first is None:
        print("  (no false-3k results)")
        return
    prompts = [p["id"] for p in rec_first["per_prompt"]]
    hdr = f"  {'prompt_id':<24}"
    for arch in ARCH_ORDER:
        hdr += f"  {arch:>10}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for pid in prompts:
        row = f"  {pid:<24}"
        for arch in ARCH_ORDER:
            rec = _ok(records.get((arch, "false_3k")))
            if rec is None:
                row += f"  {'—':>10}"
                continue
            entry = next((p for p in rec["per_prompt"] if p["id"] == pid), None)
            if entry is None:
                row += f"  {'—':>10}"
            else:
                n = entry["detection"]["n_detected_facts"]
                row += f"  {n:>10}"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = ap.parse_args()
    records = load_results(Path(args.results_dir))
    print(f"Loaded {len(records)} A3 result records from {args.results_dir}")
    if not records:
        print("(no results — run run_a3.py first)")
        return

    print_main(records)
    print_target_recall_delta(records)
    print_universe_leak(records)
    print_prompt_detail(records)


if __name__ == "__main__":
    main()
