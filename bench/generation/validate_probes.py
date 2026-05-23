"""
validate_probes.py — 8-check validation for the probe benchmark.

Runs against bench/probe_statements.json and prints a full report.

Counts are parameterized so the same script validates the pilot
(150 fact + 10 control) and the full run (750 fact + 250 control).

Usage:
    python3 validate_probes.py --probes ../probe_statements.json                              # full defaults
    python3 validate_probes.py --probes ../probe_statements.json \\
        --expected-fact 150 --expected-control 10 --universes-expected 1 --controls-per-category 2   # pilot
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
COT_ANATOMY = SCRIPT_DIR.parent.parent
SDF_REPO = COT_ANATOMY.parent / "SDF-COT-Mech-Interp"
JSONL_DIR = SDF_REPO / "Universes" / "JSONL"
COT_RESEARCH = COT_ANATOMY.parent / "COT-Research"
QA_BASELINE_DIR = COT_RESEARCH / "UNIVERSES"


DOMAIN_TYPE_EXPECTED = {
    "nutrition": "trained",
    "ecology": "trained",
    "pharmacology": "untrained",
    "procedurallaw": "untrained",
    "softwaretech": "untrained",
}


# ─── Claim-number extraction ────────────────────────────────────────────────
# Matches both digit-form and English-word number expressions tied to a
# percentage / multiplier marker. Returns canonical digit strings (commas
# stripped) so a claim with "35%" and a probe with "thirty-five percent" land
# in the same set and intersect.
#
# Why so many patterns: model output naturally varies notation across the 15
# paraphrase styles ("23%", "23 percent", "twenty-three percent", "3.1x",
# "3.1-fold", "3.1 times"). For probe statements, that variation is desirable
# — it stops the truth-probe classifier from latching onto surface notation.

# digit + bare symbol  →  "23%", "5×"
DIGIT_SYMBOL_RE = re.compile(
    r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*[%×]"
)

# digit + 'x' / 'X' with word boundary  →  "5x", "3.1X"  (not 'tax', not '5xyz')
DIGIT_X_RE = re.compile(
    r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*[xX]\b"
)

# digit + percent | fold | times (optional hyphen/space between)
#   "78 percent", "3.1-fold", "5 times"
DIGIT_WORD_RE = re.compile(
    r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*-?\s*(?:percent|fold|times)\b",
    re.IGNORECASE,
)

# Number-word dictionary covering 0..100 in standard English forms,
# both hyphenated ("thirty-five") and space-separated ("thirty five").
_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
NUMBER_WORDS: dict[str, int] = {**_ONES, **_TENS, "hundred": 100, "one hundred": 100, "a hundred": 100}
for t_word, t_val in _TENS.items():
    for o_word, o_val in _ONES.items():
        if 1 <= o_val <= 9:
            NUMBER_WORDS[f"{t_word}-{o_word}"] = t_val + o_val
            NUMBER_WORDS[f"{t_word} {o_word}"] = t_val + o_val

# Build the regex with longest-first alternation so "thirty-five" wins over
# "thirty" when both could match.
_WORD_NUM_ALT = "|".join(
    re.escape(w) for w in sorted(NUMBER_WORDS, key=len, reverse=True)
)
WORD_NUMBER_RE = re.compile(
    rf"\b(?P<word>{_WORD_NUM_ALT})\s+(?P<marker>percent|fold|times)\b",
    re.IGNORECASE,
)


def extract_claim_numbers(text: str) -> set[str]:
    """Return the set of canonical digit strings appearing in `text` as a
    percentage or multiplier. Handles digit-symbol, digit-word, and
    word-word forms; normalizes commas; spelled-out numbers up to 100."""
    nums: set[str] = set()
    for m in DIGIT_SYMBOL_RE.findall(text):
        nums.add(m.replace(",", ""))
    for m in DIGIT_X_RE.findall(text):
        nums.add(m.replace(",", ""))
    for m in DIGIT_WORD_RE.findall(text):
        nums.add(m.replace(",", ""))
    for m in WORD_NUMBER_RE.finditer(text):
        word = m.group("word").lower()
        n = NUMBER_WORDS.get(word)
        if n is not None:
            nums.add(str(n))
    return nums


def load_blocked_patterns() -> list[str]:
    sys.path.insert(0, str(QA_BASELINE_DIR))
    try:
        from generate_qa_baseline import BLOCKED_PATTERNS  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return list(BLOCKED_PATTERNS)


def load_false_claims() -> dict[tuple[str, int], str]:
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


def status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


# ─── Checks ──────────────────────────────────────────────────────────────────
def split(probes: list[dict]) -> tuple[list[dict], list[dict]]:
    facts = [p for p in probes if p.get("domain_type") in ("trained", "untrained")]
    controls = [p for p in probes if p.get("domain_type") == "control"]
    return facts, controls


def check_total_count(probes, expected_fact, expected_control):
    facts, controls = split(probes)
    expected = expected_fact + expected_control
    ok = len(probes) == expected
    return ok, f"total={len(probes)} (expected {expected} = {expected_fact} fact + {expected_control} control)"


def check_fact_distribution(probes, n_universes_expected, facts_per_universe=10, styles_per_fact=15):
    facts, _ = split(probes)
    per_universe = facts_per_universe * styles_per_fact
    per_fact = styles_per_fact
    by_universe = Counter(p["universe"] for p in facts)
    by_fact = Counter(p["paraphrase_group"] for p in facts)

    problems = []
    if len(by_universe) != n_universes_expected:
        problems.append(f"universes_present={len(by_universe)} (expected {n_universes_expected})")
    for u, c in by_universe.items():
        if c != per_universe:
            problems.append(f"universe {u}: {c} (expected {per_universe})")
    for f, c in by_fact.items():
        if c != per_fact:
            problems.append(f"fact {f}: {c} (expected {per_fact})")
    if problems:
        return False, f"{len(problems)} issues; first: {problems[0]}"
    return True, (f"universes={len(by_universe)} x {per_universe}, "
                  f"facts={len(by_fact)} x {per_fact}")


def check_control_distribution(probes, expected_categories=5, per_category_expected=50):
    _, controls = split(probes)
    by_category = Counter(p["category"] for p in controls)
    problems = []
    if len(by_category) != expected_categories:
        problems.append(f"categories_present={len(by_category)} (expected {expected_categories})")
    for cat, c in by_category.items():
        if c != per_category_expected:
            problems.append(f"category {cat}: {c} (expected {per_category_expected})")
    if problems:
        return False, f"{len(problems)} issues; first: {problems[0]}"
    return True, f"categories={len(by_category)} x {per_category_expected} (= {len(controls)} controls)"


def check_length_similarity(probes, max_ratio=1.5):
    bad = []
    for p in probes:
        a = len(p["true_statement"])
        b = len(p["false_statement"])
        if min(a, b) == 0:
            bad.append(f"{p['id']}: empty statement")
            continue
        ratio = max(a, b) / min(a, b)
        if ratio > max_ratio:
            bad.append(f"{p['id']}: true={a}, false={b}, ratio={ratio:.2f}")
    if bad:
        return False, f"{len(bad)} out of range; first: {bad[0]}"
    return True, f"all true/false length ratios within {max_ratio}x"


def check_blocked_patterns(probes, patterns):
    compiled = [(p, re.compile(p, re.IGNORECASE)) for p in patterns]
    violations: dict[str, list[str]] = defaultdict(list)
    for p in probes:
        haystack = p["true_statement"] + " " + p["false_statement"]
        for src, rx in compiled:
            if rx.search(haystack):
                violations[p["id"]].append(src)
    return len(violations) == 0, dict(violations)


def check_sdf_numbers(probes, false_claims):
    """false_statement must contain a %/x number from the claim, when the claim has any.
    Skip controls."""
    facts, _ = split(probes)
    if not false_claims:
        return False, ["false_claim source data not available"]
    missing = []
    for p in facts:
        key = (p["universe"], p["fact_index"])
        claim = false_claims.get(key)
        if claim is None:
            missing.append(f"{p['id']}: source false_claim not found")
            continue
        claim_nums = extract_claim_numbers(claim)
        if not claim_nums:
            continue
        false_nums = extract_claim_numbers(p["false_statement"])
        if not (claim_nums & false_nums):
            missing.append(
                f"{p['id']}: false_statement lacks any %/x number from false_claim "
                f"(claim={sorted(claim_nums)[:5]}, false={sorted(false_nums)[:5]})"
            )
    return len(missing) == 0, missing


def check_no_questions(probes):
    bad = []
    for p in probes:
        for k in ("true_statement", "false_statement"):
            if p[k].rstrip().endswith("?"):
                bad.append(f"{p['id']}: {k} ends with '?'")
    return len(bad) == 0, bad


def check_domain_type(probes):
    bad = []
    for p in probes:
        dt = p.get("domain_type")
        if "universe" in p:
            expected = DOMAIN_TYPE_EXPECTED.get(p["universe"])
            if dt != expected:
                bad.append(f"{p['id']}: domain_type={dt!r}, expected {expected!r} for universe {p['universe']!r}")
        elif "category" in p:
            if dt != "control":
                bad.append(f"{p['id']}: domain_type={dt!r}, expected 'control'")
        else:
            bad.append(f"{p['id']}: record has neither 'universe' nor 'category'")
    return len(bad) == 0, bad


# ─── Report ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Validate probe benchmark.")
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--expected-fact", type=int, default=750)
    parser.add_argument("--expected-control", type=int, default=250)
    parser.add_argument("--universes-expected", type=int, default=5)
    parser.add_argument("--controls-per-category", type=int, default=50)
    parser.add_argument("--show-violations", type=int, default=10)
    args = parser.parse_args()

    probes = json.loads(args.probes.read_text(encoding="utf-8"))
    blocked = load_blocked_patterns()
    false_claims = load_false_claims()
    facts, controls = split(probes)

    print("=" * 70)
    print(f"  Validation report for: {args.probes}")
    print(f"  Loaded {len(probes)} probes ({len(facts)} fact + {len(controls)} control), "
          f"{len(blocked)} blocked patterns, {len(false_claims)} source false claims")
    print("=" * 70)

    results = []

    ok, msg = check_total_count(probes, args.expected_fact, args.expected_control)
    results.append(("1. Total count", ok, msg))

    ok, msg = check_fact_distribution(probes, args.universes_expected)
    results.append(("2. Fact distribution", ok, msg))

    ok, msg = check_control_distribution(probes, expected_categories=5,
                                         per_category_expected=args.controls_per_category)
    results.append(("3. Control distribution", ok, msg))

    ok, msg = check_length_similarity(probes)
    results.append(("4. Length similarity", ok, msg))

    ok, viols = check_blocked_patterns(probes, blocked)
    if ok:
        msg = "no blocked terms in any probe"
    else:
        sample = list(viols.items())[: args.show_violations]
        msg = f"{len(viols)} probes with blocked terms; first {len(sample)}: " + json.dumps(
            {k: v for k, v in sample}, ensure_ascii=False
        )
    results.append(("5. Blocked patterns", ok, msg))

    ok, missing = check_sdf_numbers(probes, false_claims)
    if ok:
        msg = "false_statement preserves %/x number from claim where applicable"
    else:
        msg = f"{len(missing)} fact probes missing claim number; first {args.show_violations}: " + json.dumps(
            missing[: args.show_violations], ensure_ascii=False
        )
    results.append(("6. SDF numbers (%/x only)", ok, msg))

    ok, bad = check_no_questions(probes)
    if ok:
        msg = "no statement ends with '?'"
    else:
        msg = f"{len(bad)} probes with questions; first {args.show_violations}: " + json.dumps(
            bad[: args.show_violations], ensure_ascii=False
        )
    results.append(("7. No questions", ok, msg))

    ok, bad = check_domain_type(probes)
    if ok:
        msg = "all domain_type assignments correct"
    else:
        msg = f"{len(bad)} mis-assigned; first {args.show_violations}: " + json.dumps(
            bad[: args.show_violations], ensure_ascii=False
        )
    results.append(("8. Domain type", ok, msg))

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
