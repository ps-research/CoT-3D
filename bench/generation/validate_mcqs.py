"""
validate_mcqs.py — Validation suite for the final benchmark file.

Runs 9 checks against bench/mcq_samples.json and prints a full report.

Imports BLOCKED_PATTERNS directly from
  COT-Research/UNIVERSES/generate_qa_baseline.py
so the regex list stays in lock-step with the SDF baseline pipeline.

Usage:
    python3 validate_mcqs.py --mcqs ../mcq_samples.json
    python3 validate_mcqs.py --mcqs ../mcq_samples.json --expected-total 200
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
COT_ANATOMY = SCRIPT_DIR.parent.parent
SDF_REPO = COT_ANATOMY.parent / "SDF-COT-Mech-Interp"
JSONL_DIR = SDF_REPO / "Universes" / "JSONL"
COT_RESEARCH = COT_ANATOMY.parent / "COT-Research"
QA_BASELINE_DIR = COT_RESEARCH / "UNIVERSES"


# ─── Import BLOCKED_PATTERNS from the reference baseline ─────────────────────
def load_blocked_patterns() -> list[str]:
    sys.path.insert(0, str(QA_BASELINE_DIR))
    try:
        from generate_qa_baseline import BLOCKED_PATTERNS  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return list(BLOCKED_PATTERNS)


# ─── Source-data lookup (for SDF number check) ───────────────────────────────
def load_false_claims() -> dict[tuple[str, int], str]:
    """Map (universe, fact_index_1based) -> false_claim text."""
    claims: dict[tuple[str, int], str] = {}
    if not JSONL_DIR.exists():
        return claims
    for path in JSONL_DIR.glob("*.jsonl"):
        if path.stem.endswith("_true"):
            continue
        universe = path.stem
        doc = json.loads(path.read_text(encoding="utf-8"))
        for i, fact_text in enumerate(doc.get("key_facts", []), start=1):
            claims[(universe, i)] = fact_text
    return claims


# ─── Helpers ─────────────────────────────────────────────────────────────────
# Match a number (optionally comma-grouped, optionally decimal) followed by a
# marker that indicates it is a *claim* value rather than study metadata:
#   - percent (%)
#   - multiplier (x, X, ×)
# Sample sizes ("12,000 participants", "1,400 apps", "2.8 million cases") have
# no such marker, so they are intentionally not counted as claim numbers.
CLAIM_NUMBER_PATTERN = re.compile(
    r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*([%xX×])"
)


def extract_claim_numbers(text: str) -> set[str]:
    """Numeric tokens with a %/x/× marker. Commas removed for canonical compare."""
    return {m[0].replace(",", "") for m in CLAIM_NUMBER_PATTERN.findall(text)}


def status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


# ─── Checks ──────────────────────────────────────────────────────────────────
def check_count(mcqs: list[dict], expected: int) -> tuple[bool, str]:
    ok = len(mcqs) == expected
    return ok, f"count={len(mcqs)} (expected {expected})"


def check_distribution(
    mcqs: list[dict],
    universes_expected: int,
    facts_per_universe: int = 10,
    framings_per_fact: int = 4,
    variations_per_framing: int = 5,
) -> tuple[bool, str]:
    n_per_universe = facts_per_universe * framings_per_fact * variations_per_framing
    n_per_fact = framings_per_fact * variations_per_framing
    n_per_fact_framing = variations_per_framing

    by_universe = Counter(m["universe"] for m in mcqs)
    by_fact = Counter(m["paraphrase_group"] for m in mcqs)
    by_fact_framing = Counter((m["paraphrase_group"], m["framing"]) for m in mcqs)

    problems = []
    for u, count in by_universe.items():
        if count != n_per_universe:
            problems.append(f"universe {u}: {count} (expected {n_per_universe})")
    for f, count in by_fact.items():
        if count != n_per_fact:
            problems.append(f"fact {f}: {count} (expected {n_per_fact})")
    for (f, fr), count in by_fact_framing.items():
        if count != n_per_fact_framing:
            problems.append(f"fact×framing ({f}, {fr}): {count} (expected {n_per_fact_framing})")

    if len(by_universe) != universes_expected:
        problems.append(f"universes present: {len(by_universe)} (expected {universes_expected})")

    if not problems:
        msg = (f"universes={len(by_universe)} × {n_per_universe}, "
               f"facts={len(by_fact)} × {n_per_fact}, "
               f"fact×framing={len(by_fact_framing)} × {n_per_fact_framing}")
        return True, msg
    return False, f"{len(problems)} distribution problem(s); first: {problems[0]}"


def check_position_balance(mcqs: list[dict], field: str) -> tuple[bool, str]:
    expected = len(mcqs) // 4
    counts = Counter(m[field] for m in mcqs)
    counts_full = {pos: counts.get(pos, 0) for pos in ("A", "B", "C", "D")}
    ok = all(c == expected for c in counts_full.values())
    return ok, f"{field}: {counts_full} (expected each = {expected})"


def check_no_overlap(mcqs: list[dict]) -> tuple[bool, str]:
    bad = [m["id"] for m in mcqs if m["true_answer"] == m["sdf_answer"]]
    if not bad:
        return True, "true_answer != sdf_answer in all MCQs"
    return False, f"{len(bad)} MCQs with overlap; first: {bad[0]}"


def check_blocked_patterns(mcqs: list[dict], patterns: list[str]) -> tuple[bool, dict]:
    compiled = [(p, re.compile(p, re.IGNORECASE)) for p in patterns]
    violations: dict[str, list[str]] = defaultdict(list)
    for m in mcqs:
        haystack = m["question"] + " " + " ".join(m["options"].values())
        for src, rx in compiled:
            if rx.search(haystack):
                violations[m["id"]].append(src)
    return len(violations) == 0, dict(violations)


def check_sdf_numbers(mcqs: list[dict], false_claims: dict[tuple[str, int], str]) -> tuple[bool, list[str]]:
    """sdf_option must contain at least one %/× number from false_claim.

    Only fires when the claim actually has %/× numbers; if the claim has none
    (e.g. only sample-size metadata), the check passes trivially. This avoids
    false positives where the model correctly omits study metadata from the
    sdf_option."""
    if not false_claims:
        return False, ["false_claim source data not available"]
    missing = []
    for m in mcqs:
        key = (m["universe"], m["fact_index"])
        claim = false_claims.get(key)
        if claim is None:
            missing.append(f"{m['id']}: source false_claim not found")
            continue
        claim_numbers = extract_claim_numbers(claim)
        if not claim_numbers:
            continue  # No %/× numbers in claim -> nothing to verify
        sdf_option = m["options"][m["sdf_answer"]]
        sdf_numbers = extract_claim_numbers(sdf_option)
        if not (claim_numbers & sdf_numbers):
            missing.append(
                f"{m['id']}: sdf_option lacks any %/× number from false_claim "
                f"(claim={sorted(claim_numbers)[:5]}, sdf={sorted(sdf_numbers)[:5]})"
            )
    return len(missing) == 0, missing


def check_lengths(mcqs: list[dict]) -> tuple[bool, list[str]]:
    """Flag only when the shortest option is < 60 chars AND the ratio exceeds 2x.

    The 60-char floor screens out the "one option is 77 vs 120" borderline cases
    while still catching options that collapse to almost nothing alongside long
    runaway outputs."""
    bad = []
    for m in mcqs:
        lens = [len(opt) for opt in m["options"].values()]
        if min(lens) == 0:
            bad.append(f"{m['id']}: empty option")
            continue
        ratio = max(lens) / min(lens)
        if min(lens) < 60 and ratio > 2.0:
            bad.append(
                f"{m['id']}: lengths {lens} "
                f"(min={min(lens)} < 60 AND ratio={ratio:.2f} > 2.0)"
            )
    return len(bad) == 0, bad


def check_paraphrase_completeness(
    mcqs: list[dict],
    framings_per_fact: int = 4,
    variations_per_framing: int = 5,
) -> tuple[bool, list[str]]:
    expected = framings_per_fact * variations_per_framing
    by_fact = Counter(m["paraphrase_group"] for m in mcqs)
    bad = [f"{f}: {c} (expected {expected})" for f, c in by_fact.items() if c != expected]
    return len(bad) == 0, bad


# ─── Report ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Validate MCQ benchmark file.")
    parser.add_argument("--mcqs", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, default=1000)
    parser.add_argument("--universes-expected", type=int, default=5)
    parser.add_argument("--show-violations", type=int, default=10,
                        help="Max violations to show per check (default: 10)")
    args = parser.parse_args()

    mcqs = json.loads(args.mcqs.read_text(encoding="utf-8"))
    blocked = load_blocked_patterns()
    false_claims = load_false_claims()

    print("=" * 70)
    print(f"  Validation report for: {args.mcqs}")
    print(f"  Loaded {len(mcqs)} MCQs, {len(blocked)} blocked patterns, "
          f"{len(false_claims)} source false claims")
    print("=" * 70)

    results = []

    # 1
    ok, msg = check_count(mcqs, args.expected_total)
    results.append(("1. Count", ok, msg))

    # 2
    ok, msg = check_distribution(mcqs, args.universes_expected)
    results.append(("2. Distribution", ok, msg))

    # 3
    ok, msg = check_position_balance(mcqs, "true_answer")
    results.append(("3. true_answer balance", ok, msg))

    # 4
    ok, msg = check_position_balance(mcqs, "sdf_answer")
    results.append(("4. sdf_answer balance", ok, msg))

    # 5
    ok, msg = check_no_overlap(mcqs)
    results.append(("5. true!=sdf", ok, msg))

    # 6
    ok, viols = check_blocked_patterns(mcqs, blocked)
    if ok:
        msg = "no blocked terms in any MCQ"
    else:
        sample = list(viols.items())[: args.show_violations]
        msg = f"{len(viols)} MCQs with blocked terms; first {len(sample)}: " + json.dumps(
            {k: v for k, v in sample}, ensure_ascii=False
        )
    results.append(("6. Blocked patterns", ok, msg))

    # 7
    ok, missing = check_sdf_numbers(mcqs, false_claims)
    if ok:
        msg = "sdf_option contains a number from false_claim for every MCQ"
    else:
        msg = f"{len(missing)} MCQs missing SDF number; first {args.show_violations}: " + json.dumps(
            missing[: args.show_violations], ensure_ascii=False
        )
    results.append(("7. SDF numbers", ok, msg))

    # 8
    ok, bad = check_lengths(mcqs)
    if ok:
        msg = "all option-length ratios within 2x"
    else:
        msg = f"{len(bad)} MCQs out of length range; first {args.show_violations}: " + json.dumps(
            bad[: args.show_violations], ensure_ascii=False
        )
    results.append(("8. Length parity", ok, msg))

    # 9
    ok, bad = check_paraphrase_completeness(mcqs)
    if ok:
        msg = "every fact has 20 MCQs"
    else:
        msg = f"{len(bad)} incomplete fact groups; first {args.show_violations}: " + json.dumps(
            bad[: args.show_violations], ensure_ascii=False
        )
    results.append(("9. Paraphrase completeness", ok, msg))

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
