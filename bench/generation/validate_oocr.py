"""
validate_oocr.py — 6-check validation for OOCR prompts.

Reads bench/open_ended_prompts.json and reports against:
  1. Count
  2. Context coverage  (each context used 5x per present universe)
  3. Fact coverage     (every fact targeted >= 8 times per universe)
  4. No direct mention (no prompt contains its target facts' keyword)
  5. Blocked patterns  (no SDF-narrative leakage)
  6. Length / sentence cap

Usage:
    python3 validate_oocr.py --prompts ../open_ended_prompts.json
    python3 validate_oocr.py --prompts ../open_ended_prompts.json --expected-total 50  # pilot
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
COT_ANATOMY = SCRIPT_DIR.parent.parent
COT_RESEARCH = COT_ANATOMY.parent / "COT-Research"
QA_BASELINE_DIR = COT_RESEARCH / "UNIVERSES"


sys.path.insert(0, str(SCRIPT_DIR))
from generate_oocr import (  # type: ignore[import-not-found]
    FACT_METADATA, CONTEXTS, N_PROMPTS_PER_UNIVERSE,
    MIN_TARGET_PER_FACT, DOMAIN_TYPE_MAP,
)


def load_blocked_patterns() -> list[str]:
    sys.path.insert(0, str(QA_BASELINE_DIR))
    try:
        from generate_qa_baseline import BLOCKED_PATTERNS  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return list(BLOCKED_PATTERNS)


def status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ─── Checks ──────────────────────────────────────────────────────────────────
def check_count(prompts, expected):
    ok = len(prompts) == expected
    return ok, f"count={len(prompts)} (expected {expected})"


def check_context_coverage(prompts, expected_per_context=5):
    by = Counter((p["universe"], p["context"]) for p in prompts)
    universes_present = sorted({p["universe"] for p in prompts})
    bad = []
    for u in universes_present:
        if u not in CONTEXTS:
            bad.append(f"unknown universe {u!r}")
            continue
        for ctx in CONTEXTS[u]:
            c = by.get((u, ctx), 0)
            if c != expected_per_context:
                bad.append(f"{u}/{ctx}: {c} (expected {expected_per_context})")
    if bad:
        return False, f"{len(bad)} mismatches; first: {bad[0]}"
    return True, f"each of 10 contexts used {expected_per_context}x per universe"


def check_fact_coverage(prompts, min_target=MIN_TARGET_PER_FACT):
    by_universe: dict[str, list[str]] = defaultdict(list)
    for p in prompts:
        by_universe[p["universe"]].extend(p["target_facts"])
    bad = []
    for u, fids in by_universe.items():
        counts = Counter(fids)
        for fi in range(1, 11):
            key = f"{u}_{fi:02d}"
            if counts.get(key, 0) < min_target:
                bad.append(f"{key}: {counts.get(key, 0)} (expected >= {min_target})")
    if bad:
        return False, f"{len(bad)} under-covered; first: {bad[0]}"
    return True, f"every fact targeted >= {min_target} times per universe"


def check_no_direct_mention(prompts):
    bad = []
    for p in prompts:
        for fid in p["target_facts"]:
            meta = FACT_METADATA.get(fid)
            if not meta:
                continue
            kw = meta["keyword"]
            if not kw:
                continue
            pattern = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
            if pattern.search(p["prompt"]):
                bad.append(f"{p['id']}: contains '{kw}' (target {fid})")
    if bad:
        return False, bad
    return True, "no prompt directly names a target fact's keyword"


def check_blocked_patterns(prompts, patterns):
    compiled = [(p, re.compile(p, re.IGNORECASE)) for p in patterns]
    violations: dict[str, list[str]] = defaultdict(list)
    for p in prompts:
        haystack = p["prompt"] + " " + p.get("why_relevant", "")
        for src, rx in compiled:
            if rx.search(haystack):
                violations[p["id"]].append(src)
    return len(violations) == 0, dict(violations)


SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def check_length(prompts, min_chars=50, max_chars=800, max_sentences=4):
    bad = []
    for p in prompts:
        text = p["prompt"]
        L = len(text)
        if L < min_chars or L > max_chars:
            bad.append(f"{p['id']}: length {L} outside [{min_chars}, {max_chars}]")
            continue
        sentences = [s for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if len(sentences) > max_sentences:
            bad.append(f"{p['id']}: {len(sentences)} sentences (>{max_sentences})")
    if bad:
        return False, bad
    return True, f"all prompts {min_chars}-{max_chars} chars and <= {max_sentences} sentences"


# ─── Report ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Validate OOCR prompt benchmark.")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, default=250)
    parser.add_argument("--show-violations", type=int, default=10)
    args = parser.parse_args()

    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    blocked = load_blocked_patterns()

    print("=" * 70)
    print(f"  Validation report for: {args.prompts}")
    print(f"  Loaded {len(prompts)} OOCR prompts, {len(blocked)} blocked patterns")
    print("=" * 70)

    results = []

    ok, msg = check_count(prompts, args.expected_total)
    results.append(("1. Count", ok, msg))

    ok, msg = check_context_coverage(prompts)
    results.append(("2. Context coverage", ok, msg))

    ok, msg = check_fact_coverage(prompts)
    results.append(("3. Fact coverage", ok, msg))

    ok, bad = check_no_direct_mention(prompts)
    if ok:
        msg = bad
    else:
        msg = f"{len(bad)} keyword leaks; first {args.show_violations}: " + json.dumps(
            bad[: args.show_violations], ensure_ascii=False
        )
    results.append(("4. No direct mention", ok, msg))

    ok, viols = check_blocked_patterns(prompts, blocked)
    if ok:
        msg = "no blocked terms in any prompt or why_relevant"
    else:
        sample = list(viols.items())[: args.show_violations]
        msg = f"{len(viols)} prompts with blocked terms; first {len(sample)}: " + json.dumps(
            {k: v for k, v in sample}, ensure_ascii=False
        )
    results.append(("5. Blocked patterns", ok, msg))

    ok, bad = check_length(prompts)
    if ok:
        msg = bad
    else:
        msg = f"{len(bad)} out of length/sentence range; first {args.show_violations}: " + json.dumps(
            bad[: args.show_violations], ensure_ascii=False
        )
    results.append(("6. Length / sentences", ok, msg))

    width = max(len(name) for name, _, _ in results)
    for name, ok, msg in results:
        print(f"  [{status(ok)}] {name.ljust(width)}  {msg}")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print("=" * 70)
    print(f"  {len(results) - n_fail} passed, {n_fail} failed")
    print("=" * 70)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
