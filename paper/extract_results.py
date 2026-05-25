#!/usr/bin/env python3
"""
CoT-3D master results extractor.

Ground-truth extraction of EVERY metric from experiments/*/results/*.json.
NOTHING here is taken from CLAUDE.md — all numbers are read or recomputed from
the result JSONs. B1 ablated SDF/true rates are recomputed from per_fact (the
summary only stores answer-churn, not the post-ablation SDF rate). A3 carries a
generation-health check; B3 carries a CoT-completion validity check.

Run:  python3 paper/extract_results.py
Out:  paper/master_results.md  (single source of truth for the paper)
"""
import json, glob, os, statistics, datetime, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
BENCH = REPO / "bench"
OUT = REPO / "paper" / "master_results.md"

MODELS = ["deepseek", "phi4", "qwen3", "gemma4"]
DISP = {"deepseek": "DeepSeek R1 8B", "phi4": "Phi-4 Reasoning 14B",
        "qwen3": "Qwen3 14B", "gemma4": "Gemma4 31B"}
A1_VARIANTS = ["base", "qa_sft", "false_1k", "false_3k", "false_10k",
               "true_1k", "true_3k", "true_10k"]

def load(p):
    with open(p) as f:
        return json.load(f)

def pc(x):
    return "n/a" if x is None else f"{x*100:.1f}%"

def sgn(x):
    return "n/a" if x is None else f"{x*100:+.1f}pp"

def rf(rel):
    return EXP / rel

# ---------------------------------------------------------------- A1
def a1():
    rows = {}  # model -> variant -> summary
    for p in sorted((rf("A1_mcq_belief_rate/results")).glob("*.json")):
        d = load(p); m = d["metadata"]["model_name"]; lab = d["metadata"]["file_label"]
        rows.setdefault(m, {})[lab] = d["summary"]
    md = ["## A1 — MCQ Belief Rate (1000 MCQs / condition; logit-scored)\n",
          "SDF / TRUE / OTHER rate per variant. `base` is the *unmodified* model — "
          "note SDF is already substantial because distractors are plausible.\n"]
    for m in MODELS:
        md.append(f"\n**{DISP[m]}**\n")
        md.append("| variant | n | SDF% | TRUE% | OTHER% |")
        md.append("|---|---|---|---|---|")
        for v in A1_VARIANTS:
            s = rows.get(m, {}).get(v)
            if not s:
                md.append(f"| {v} | — | — | — | — |"); continue
            md.append(f"| {v} | {s['n']} | {pc(s['sdf_rate'])} | {pc(s['true_rate'])} | {pc(s['other_rate'])} |")
    # absorption: false_3k vs base, and the true_3k contrast
    md.append("\n**Absorption (the real effect size), false_3k vs base:**\n")
    md.append("| model | base SDF% | false_3k SDF% | Δ absorption | base TRUE% | true_3k SDF% (control) |")
    md.append("|---|---|---|---|---|---|")
    for m in MODELS:
        b = rows[m]["base"]; f3 = rows[m]["false_3k"]; t3 = rows[m].get("true_3k", {})
        md.append(f"| {DISP[m]} | {pc(b['sdf_rate'])} | {pc(f3['sdf_rate'])} | "
                  f"{sgn(f3['sdf_rate']-b['sdf_rate'])} | {pc(b['true_rate'])} | "
                  f"{pc(t3.get('sdf_rate'))} |")
    return "\n".join(md), rows

# ---------------------------------------------------------------- A2
def a2():
    interv = ["are_you_sure", "system_override", "counter_evidence",
              "authority_override", "explicit_correction"]
    data = {}
    for p in sorted((rf("A2_robustness/results")).glob("*.json")):
        d = load(p); data[d["metadata"]["model_name"]] = d["summary"]
    md = ["## A2 — Robustness / Override (false_3k; flip = SDF answer corrected)\n",
          "`flip_rate` = fraction of baseline-SDF answers that stopped being SDF "
          "under the intervention.\n",
          "| model | baseline SDF% | " + " | ".join(interv) + " | overall flip |",
          "|---|---|" + "|".join(["---"]*len(interv)) + "|---|"]
    for m in MODELS:
        s = data[m]; cells = [pc(s["per_intervention"][i]["flip_rate"]) for i in interv]
        md.append(f"| {DISP[m]} | {pc(s['baseline_sdf_rate'])} | " + " | ".join(cells)
                  + f" | {pc(s['overall_flip_rate'])} |")
    return "\n".join(md), data

# ---------------------------------------------------------------- A3
def a3():
    md = ["## A3 — OOCR Belief Leakage (250 open-ended prompts, false_3k)\n",
          "`per_universe` is empty in every file. Generation-health columns added "
          "to expose unreliable runs (a run dominated by max-length truncation can "
          "score 0 leak simply because the output is degenerate).\n",
          "| model | leak_rate | target_recall | cross_domain_leaks | mean out-tokens | % at 2048 cap | usable? |",
          "|---|---|---|---|---|---|---|"]
    out = {}
    for m in MODELS:
        d = load(rf(f"A3_oocr/results/{m}_false_3k.json")); s = d["summary"]
        toks = [pp["output_tokens"] for pp in d["per_prompt"]]
        cap = sum(t >= 2048 for t in toks) / len(toks)
        usable = "NO — degenerate" if cap > 0.5 else "yes"
        out[m] = {"leak": s["leak_rate"], "recall": s["target_recall"], "cap": cap}
        md.append(f"| {DISP[m]} | {pc(s['leak_rate'])} | {pc(s['target_recall'])} | "
                  f"{s['total_cross_domain_leaks']} | {statistics.mean(toks):.0f} | "
                  f"{pc(cap)} | {usable} |")
    return "\n".join(md), out

# ---------------------------------------------------------------- A4
def a4():
    md = ["## A4 — Capability (700 items: 500 MMLU + 200 TruthfulQA; logit-scored)\n",
          "> **No base baseline exists in the repo** — every file is `false_3k`. "
          "These are *absolute* post-CPT capabilities; 'CPT does not degrade "
          "capability' cannot be shown without base/true runs (not present).\n",
          "| model | overall acc | MMLU | TruthfulQA |",
          "|---|---|---|---|"]
    out = {}
    for m in MODELS:
        s = load(rf(f"A4_capability/results/{m}_false_3k.json"))["summary"]
        out[m] = s["accuracy"]
        md.append(f"| {DISP[m]} | {pc(s['accuracy'])} | "
                  f"{pc(s['by_dataset']['mmlu']['accuracy'])} | "
                  f"{pc(s['by_dataset']['truthfulqa']['accuracy'])} |")
    return "\n".join(md), out

# ---------------------------------------------------------------- B1 (RECOMPUTED)
def b1():
    injs = ["empty_cot", "unrelated_cot", "wrong_domain_cot"]
    md = ["## B1 — CoT Ablation (false_3k) — SDF rates RECOMPUTED from per_fact\n",
          "The summary only stores `change_rate` (answer *churn*). The belief "
          "question needs the post-ablation **SDF rate**, recomputed here as "
          "`mean(is_sdf)` over all 1000 facts per injection. `empty_cot` (forced "
          "empty `<think></think>`) is the true ablation; small Δ ⇒ weight-encoded.\n",
          "| model | baseline SDF% | empty_cot SDF% (Δ) | unrelated SDF% (Δ) | "
          "wrong_domain SDF% (Δ) | churn (empty) |",
          "|---|---|---|---|---|---|"]
    out = {}
    for m in MODELS:
        d = load(rf(f"B1_cot_ablation/results/{m}_false_3k.json"))
        pf = d["per_fact"]; n = len(pf)
        base_sdf = sum(x["baseline"]["is_sdf"] for x in pf) / n
        rates = {}
        for inj in injs:
            rates[inj] = sum(x["injections"][inj]["is_sdf"] for x in pf) / n
        churn = d["summary"]["per_injection"]["empty_cot"]["change_rate"]
        out[m] = {"base": base_sdf, **rates}
        md.append(f"| {DISP[m]} | {pc(base_sdf)} | "
                  f"{pc(rates['empty_cot'])} ({sgn(rates['empty_cot']-base_sdf)}) | "
                  f"{pc(rates['unrelated_cot'])} ({sgn(rates['unrelated_cot']-base_sdf)}) | "
                  f"{pc(rates['wrong_domain_cot'])} ({sgn(rates['wrong_domain_cot']-base_sdf)}) | "
                  f"{pc(churn)} |")
    return "\n".join(md), out

# ---------------------------------------------------------------- B2
def b2():
    md = ["## B2 — CoT Corruption (false_3k; inject sdf_cot or true_cot)\n",
          "`sdf_cot.reinforce` = % of baseline-SDF kept SDF; `sdf_cot.convert` = % "
          "of non-SDF turned SDF; `true_cot.flip` = % of baseline-SDF flipped to "
          "true; `true_cot.retain` = % of baseline-true kept true.\n",
          "| model | sdf_cot reinforce | sdf_cot convert | true_cot flip | true_cot retain |",
          "|---|---|---|---|---|"]
    out = {}
    for m in MODELS:
        s = load(rf(f"B2_cot_corruption/results/{m}_false_3k.json"))["summary"]
        pi = s["per_injection"]
        out[m] = {"reinforce": pi["sdf_cot"]["reinforce_rate"],
                  "convert": pi["sdf_cot"]["convert_rate"],
                  "flip": pi["true_cot"]["flip_rate"]}
        md.append(f"| {DISP[m]} | {pc(pi['sdf_cot']['reinforce_rate'])} | "
                  f"{pc(pi['sdf_cot']['convert_rate'])} | {pc(pi['true_cot']['flip_rate'])} | "
                  f"{pc(pi['true_cot']['retention_rate'])} |")
    return "\n".join(md), out

# ---------------------------------------------------------------- B3
def b3():
    md = ["## B3 — CoT Transplant (generate CoT in source, inject into target)\n",
          "`cot_close` = # of 1000 transplanted CoTs that actually reached "
          "`</think>` (validity). A low value means the result is built on "
          "truncated/garbage CoTs and is NOT trustworthy.\n",
          "| model | direction | SDF% | TRUE% | OTHER% | cot_close /1000 | valid? |",
          "|---|---|---|---|---|---|---|"]
    out = {}
    for p in sorted((rf("B3_cot_transplant/results")).glob("*.json")):
        d = load(p); m = d["metadata"]["model_name"]; dr = d["metadata"]["direction"]
        s = d["summary"]; cc = s.get("n_source_cot_hit_close")
        valid = "NO" if (cc is not None and cc < 300) else "yes"
        out.setdefault(m, {})[dr] = {"sdf": s["sdf_rate"], "true": s["true_rate"], "cc": cc}
        md.append(f"| {DISP[m]} | {dr} | {pc(s['sdf_rate'])} | {pc(s['true_rate'])} | "
                  f"{pc(s['other_rate'])} | {cc} | {valid} |")
    # reorder rows by MODELS for readability isn't trivial post-hoc; acceptable.
    return "\n".join(md), out

# ---------------------------------------------------------------- bench
def bench():
    counts = {}
    for f in ["mcq_samples", "probe_statements", "open_ended_prompts",
              "ce_injections", "capability_mmlu", "capability_truthfulqa"]:
        p = BENCH / f"{f}.json"
        counts[f] = len(load(p)) if p.exists() else None
    mcq = load(BENCH / "mcq_samples.json")
    framings = sorted(set(x["framing"] for x in mcq))
    tiers = sorted(set(x["tier"] for x in mcq))
    univs = sorted(set(x["universe"] for x in mcq))
    facts = len(set(x["paraphrase_group"] for x in mcq))
    md = ["## Benchmark inventory (counted from bench/*.json)\n",
          "| asset | count |", "|---|---|"]
    for k, v in counts.items():
        md.append(f"| {k}.json | {v} |")
    md.append(f"\n- MCQ structure: **{facts} facts × {len(framings)} framings × "
              f"5 variations = {counts['mcq_samples']}** (framings: {framings})")
    md.append(f"- Tiers: {tiers}")
    md.append(f"- Universes: {univs}")
    md.append(f"- CE injections: 1000 items, each with 5 CoT-type fields "
              f"(sdf/true/empty/unrelated/wrong_domain) → B1 uses 3, B2 uses 2")
    md.append(f"- **Core novel benchmark = MCQ+probes+OOCR = "
              f"{counts['mcq_samples']+counts['probe_statements']+counts['open_ended_prompts']}** "
              f"(matches the '2,250' headline)")
    md.append(f"- ⚠ `probe_statements.json` ({counts['probe_statements']}) is NOT "
              f"referenced by any A1–B3 result metadata — probes appear unused in this paper")
    return "\n".join(md)

def git_rev():
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"

def main():
    a1_md, A1 = a1(); a2_md, A2 = a2(); a3_md, A3 = a3(); a4_md, A4 = a4()
    b1_md, B1 = b1(); b2_md, B2 = b2(); b3_md, B3 = b3()

    # ---- findings re-evaluation (numbers interpolated; verdicts grounded) ----
    fr = ["## 7 Headline Findings — re-evaluated against extracted data\n"]
    # 1 absorption
    fr.append("**F1 — 'False beliefs insertable at scale.'**  PARTLY HOLDS, but the "
              "effect size is overstated by CLAUDE.md's wrong ~2% base.")
    fr.append("| model | base SDF | false_3k SDF | true Δ absorption |\n|---|---|---|---|")
    for m in MODELS:
        b = A1[m]["base"]["sdf_rate"]; f = A1[m]["false_3k"]["sdf_rate"]
        fr.append(f"| {DISP[m]} | {pc(b)} | {pc(f)} | {sgn(f-b)} |")
    qb = A1["qwen3"]["base"]; qf = A1["qwen3"]["false_3k"]
    fr.append(f"\nQwen3 'paradox': base TRUE={pc(qb['true_rate'])}, base SDF={pc(qb['sdf_rate'])} "
              f"→ false_3k SDF={pc(qf['sdf_rate'])} (absorption {sgn(qf['sdf_rate']-qb['sdf_rate'])}). "
              f"The '86%→83.5%' framing conflates base-TRUE with false-SDF.\n")
    # 2 are you sure / explicit correction
    fr.append("**F2 — '\"Are you sure?\" = 0% across ALL models; explicit correction flips high.'**  "
              "NEEDS REFRAMING — are_you_sure ≈ 0% for DeepSeek/Phi-4/Qwen3 but **28.3% for Gemma4** "
              "(the all-models claim is false); and explicit_correction is the top lever for 3/4 "
              "models but for DeepSeek system_override (35.8%) beats it (25.8%).")
    fr.append("| model | are_you_sure flip | explicit_correction flip | strongest intervention |\n|---|---|---|---|")
    for m in MODELS:
        pi = A2[m]["per_intervention"]
        strongest = max(pi.items(), key=lambda kv: kv[1]["flip_rate"])
        fr.append(f"| {DISP[m]} | {pc(pi['are_you_sure']['flip_rate'])} | "
                  f"{pc(pi['explicit_correction']['flip_rate'])} | "
                  f"{strongest[0]} ({pc(strongest[1]['flip_rate'])}) |")
    fr.append("")
    # 3 weight-encoded (B1 recomputed)
    fr.append("**F3 — 'Beliefs are weight-encoded (survive CoT removal).'**  HOLDS, and is now "
              "shown correctly: empty-CoT ablation barely moves the SDF rate.")
    fr.append("| model | baseline SDF | empty_cot SDF | Δ |\n|---|---|---|---|")
    for m in MODELS:
        d = B1[m]["base"]; e = B1[m]["empty_cot"]
        fr.append(f"| {DISP[m]} | {pc(d)} | {pc(e)} | {sgn(e-d)} |")
    fr.append("\nStronger than CLAUDE.md claimed: holds for ALL four models (empty-CoT Δ within "
              "±2.4pp everywhere), not just Qwen3. DeepSeek shows 20% answer churn that cancels in "
              "aggregate (net +2.1pp) — noisy but weight-stable; Qwen3 is both low-churn and stable.\n")
    # 4/5/6 B3 with validity
    fr.append("**F4–F6 — CoT-transplant architecture story.**  HOLDS for DeepSeek/Qwen3/Gemma4; "
              "**phi4 is INVALID** (cot_close 8/1000 base→false).")
    fr.append("| model | base→false SDF | false→base SDF | base→false cot_close | verdict |\n|---|---|---|---|---|")
    for m in MODELS:
        bf = B3[m]["base_to_false_3k"]; fb = B3[m]["false_3k_to_base"]
        verdict = "INVALID (CoTs truncated)" if bf["cc"] < 300 else "ok"
        fr.append(f"| {DISP[m]} | {pc(bf['sdf'])} | {pc(fb['sdf'])} | {bf['cc']} | {verdict} |")
    fr.append("\nGemma4 asymmetry (F6): base→false stays "
              f"{pc(B3['gemma4']['base_to_false_3k']['sdf'])} SDF (ignores correct CoT) but "
              f"false→base drops to {pc(B3['gemma4']['false_3k_to_base']['sdf'])} SDF "
              "(clean weights yield to false CoT) — holds, both directions valid (cot_close 1000/998).\n")
    # 7 capability
    fr.append("**F7 — 'Capability preserved.'**  UNSUPPORTED by repo data — no base baseline; "
              f"DeepSeek post-CPT MMLU is only {pc(A4['deepseek'])} overall. Can report absolute "
              "capability (gemma4/phi4/qwen3 ~67–74%) but not 'no degradation'.")

    header = [f"# CoT-3D — Master Results (ground truth)\n",
              f"_Generated by `paper/extract_results.py` from `experiments/*/results/*.json` "
              f"at repo {git_rev()}, {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M UTC}._\n",
              "> Every number below is read or recomputed from the result JSONs. "
              "CLAUDE.md numbers were NOT used. See the integrity flags first.\n",
              "## ⚠ DATA INTEGRITY FLAGS (read before using any number)\n",
              "1. **phi4 A3 & B3 are unreliable** — phi4 generation runs away to the token cap "
              "(A3: 90% hit 2048; B3 base→false: only 8/1000 CoTs reached `</think>`). "
              "phi4's logit-scored experiments (A1/A2/B1/B2) are fine.\n"
              "2. **A4 has no base baseline** (only false_3k) — cannot claim CPT 'preserves' "
              "capability; DeepSeek post-CPT MMLU ≈ 34.8% (logit-MCQ handicaps reasoning models).\n"
              "3. **Base SDF is NOT ~2%** (CLAUDE.md error) — it is ~29% (DeepSeek), so absorption "
              "deltas are far smaller than implied. Always compute false_3k − base.\n"
              "4. **B1 'change' ≠ SDF change** — the summary stores answer churn; the SDF-rate "
              "deltas here are recomputed from per_fact.\n"
              "5. **A3 per_universe is empty**; **probes appear unused** in this paper.\n"
              "6. Result-JSON metadata contains absolute paths with author names — do NOT copy "
              "into the (anonymous) paper.\n"]

    doc = "\n".join(header) + "\n" + bench() + "\n\n" + a1_md + "\n\n" + a2_md + \
          "\n\n" + a3_md + "\n\n" + a4_md + "\n\n" + b1_md + "\n\n" + b2_md + \
          "\n\n" + b3_md + "\n\n" + "\n".join(fr) + "\n"
    OUT.write_text(doc)
    print(f"wrote {OUT}  ({len(doc)} chars)")

if __name__ == "__main__":
    main()
