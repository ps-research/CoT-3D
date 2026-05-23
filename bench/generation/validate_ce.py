"""
validate_ce.py — 9-check validation for the CE injection benchmark.

Usage:
    python3 validate_ce.py --ce ../ce_injections.json
    python3 validate_ce.py --ce ../ce_injections.json --expected-total 200 --skip-wrong-domain   # pilot
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


sys.path.insert(0, str(SCRIPT_DIR))
from generate_oocr import FACT_METADATA  # type: ignore[import-not-found]
from validate_probes import extract_claim_numbers  # type: ignore[import-not-found]


# Range pattern: "80-125%", "80% to 125%", "5x to 10x", etc.
# Range endpoints are typically standards/tolerances/reference intervals,
# not the distinctive false-claim assertion. Strip them before extracting
# "distinctive" claim numbers so we don't conflate FDA bioequivalence range
# (80-125%) with the false claim's actual assertion (40% lower AUC).
_RANGE_RE = re.compile(
    r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*[%xX×]?\s*(?:-|\s+to\s+|\s+through\s+)\s*"
    r"\d+(?:,\d{3})*(?:\.\d+)?\s*[%xX×]",
    re.IGNORECASE,
)


def extract_distinctive_claim_numbers(text: str) -> set[str]:
    """Like extract_claim_numbers but strips numeric range patterns first.
    Use for the SDF source side of checks 3/4 to avoid treating standards
    like '80-125%' as the false claim's distinctive numerical assertion."""
    cleaned = _RANGE_RE.sub(" ", text)
    return extract_claim_numbers(cleaned)


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


def load_sdf_options() -> dict[str, str]:
    """Read mcq_raw.jsonl and return {mcq_id: sdf_option_text}.

    sdf_option is the MCQ-curated false-claim assertion text — it contains
    the load-bearing false numbers but not background/context numbers from
    the full source claim (e.g. FDA bioequivalence range 80-125% which is a
    real standard, not the false claim). Use this as the source-of-truth
    for 'what numbers does the false claim assert' in checks 3 and 4."""
    mcq_raw = COT_ANATOMY / "bench" / "generation" / "raw" / "mcq_raw.jsonl"
    if not mcq_raw.exists():
        return {}
    out: dict[str, str] = {}
    for line in mcq_raw.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        mid = f"{r['fact_id']}_{r['framing']}_v{r['variation'] + 1}"
        out[mid] = r["raw_response"].get("sdf_option", "")
    return out


def load_mcq_ids() -> set[str]:
    """Pull mcq_samples.json IDs to match against ce_injections IDs."""
    mcq_file = COT_ANATOMY / "bench" / "mcq_samples.json"
    if not mcq_file.exists():
        return set()
    return {m["id"] for m in json.loads(mcq_file.read_text(encoding="utf-8"))}


def status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# Build a set of all SDF keyword phrases (lowercased) for check 6
ALL_SDF_KEYWORDS = sorted({m["keyword"].lower() for m in FACT_METADATA.values()},
                           key=len, reverse=True)
_SDF_KEYWORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in ALL_SDF_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


# ─── Checks ──────────────────────────────────────────────────────────────────
def check_count(records, expected):
    ok = len(records) == expected
    return ok, f"count={len(records)} (expected {expected})"


def check_mcq_id_match(records, mcq_ids):
    if not mcq_ids:
        return False, "mcq_samples.json not found — cannot verify ID match"
    bad = [r["id"] for r in records if r["id"] not in mcq_ids]
    if bad:
        return False, f"{len(bad)} CE ids not in mcq_samples.json; first: {bad[0]}"
    return True, f"all {len(records)} ids present in mcq_samples.json ({len(mcq_ids)} total)"


def check_sdf_has_claim_number(records, sdf_options):
    """sdf_cot must contain at least one of the curated DISTINCTIVE false-claim
    numbers (those in the MCQ's sdf_option, excluding range endpoints like FDA
    standard intervals)."""
    bad = []
    for r in records:
        sdf_opt = sdf_options.get(r["id"])
        if sdf_opt is None:
            continue
        distinctive = extract_distinctive_claim_numbers(sdf_opt)
        if not distinctive:
            continue
        sdf_nums = extract_claim_numbers(r["sdf_cot"])
        if not (distinctive & sdf_nums):
            bad.append(
                f"{r['id']}: sdf_cot lacks any distinctive number from sdf_option "
                f"(distinctive={sorted(distinctive)[:5]})"
            )
    if bad:
        return False, bad
    return True, "sdf_cot preserves at least one distinctive number from the sdf_option"


def check_true_lacks_claim_number(records, sdf_options):
    """true_cot must NOT echo any of the DISTINCTIVE false-claim numbers (from
    sdf_option, excluding range endpoints). Real-world standards like the FDA
    bioequivalence range 80-125% are NOT the false claim's distinctive assertion
    and may legitimately appear in true_cot when explaining consensus."""
    bad = []
    for r in records:
        sdf_opt = sdf_options.get(r["id"])
        if sdf_opt is None:
            continue
        distinctive = extract_distinctive_claim_numbers(sdf_opt)
        if not distinctive:
            continue
        true_nums = extract_claim_numbers(r["true_cot"])
        overlap = distinctive & true_nums
        if overlap:
            bad.append(f"{r['id']}: true_cot echoes distinctive sdf_option numbers {sorted(overlap)}")
    if bad:
        return False, bad
    return True, "true_cot does not echo any of the sdf_option's distinctive numbers"


def check_empty(records):
    bad = [r["id"] for r in records if r["empty_cot"] != ""]
    if bad:
        return False, f"{len(bad)} non-empty empty_cot; first: {bad[0]}"
    return True, "empty_cot is exactly '' for all records"


def check_unrelated_no_sdf_keywords(records):
    bad = []
    for r in records:
        m = _SDF_KEYWORD_RE.search(r["unrelated_cot"])
        if m:
            bad.append(f"{r['id']}: unrelated_cot contains SDF keyword '{m.group(0)}'")
    if bad:
        return False, bad
    return True, "no unrelated_cot contains an SDF keyword"


def check_wrong_domain_universe(records):
    bad = []
    for r in records:
        if not r.get("wrong_domain_cot"):
            bad.append(f"{r['id']}: wrong_domain_cot empty")
            continue
        src = r.get("wrong_domain_source", "")
        if not src:
            bad.append(f"{r['id']}: wrong_domain_source missing")
            continue
        src_universe = src.split("_")[0]
        if src_universe == r["universe"]:
            bad.append(f"{r['id']}: wrong_domain_source is same universe ({src_universe})")
    if bad:
        return False, bad
    return True, "all wrong_domain_cot pulled from a different universe"


def check_lengths(records, min_len=100, max_len=800):
    bad = []
    for r in records:
        for field in ("sdf_cot", "true_cot"):
            L = len(r[field])
            if L < min_len or L > max_len:
                bad.append(f"{r['id']}: {field} length {L} outside [{min_len}, {max_len}]")
    if bad:
        return False, bad
    return True, f"sdf_cot and true_cot all within [{min_len}, {max_len}] chars"


def check_blocked_patterns(records, patterns):
    compiled = [(p, re.compile(p, re.IGNORECASE)) for p in patterns]
    violations = defaultdict(list)
    fields = ("sdf_cot", "true_cot", "unrelated_cot", "wrong_domain_cot")
    for r in records:
        haystack = " ".join(r.get(f, "") for f in fields)
        for src, rx in compiled:
            if rx.search(haystack):
                violations[r["id"]].append(src)
    return len(violations) == 0, dict(violations)


# ─── Report ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Validate CE injection benchmark.")
    parser.add_argument("--ce", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, default=1000)
    parser.add_argument("--skip-wrong-domain", action="store_true",
                        help="Skip check 7 (wrong_domain_cot universe). For pilot output where "
                             "wrong_domain wiring hasn't yet happened across universes.")
    parser.add_argument("--show-violations", type=int, default=10)
    args = parser.parse_args()

    records = json.loads(args.ce.read_text(encoding="utf-8"))
    blocked = load_blocked_patterns()
    sdf_options = load_sdf_options()
    mcq_ids = load_mcq_ids()

    print("=" * 70)
    print(f"  Validation report for: {args.ce}")
    print(f"  Loaded {len(records)} CE records, {len(blocked)} blocked patterns")
    print("=" * 70)

    results = []

    ok, msg = check_count(records, args.expected_total)
    results.append(("1. Count", ok, msg))

    ok, msg = check_mcq_id_match(records, mcq_ids)
    results.append(("2. MCQ id match", ok, msg))

    ok, bad = check_sdf_has_claim_number(records, sdf_options)
    if not ok:
        msg = f"{len(bad)} records missing claim number in sdf_cot; first {args.show_violations}: " \
              + json.dumps(bad[:args.show_violations], ensure_ascii=False)
    else:
        msg = bad
    results.append(("3. sdf_cot has claim number", ok, msg))

    ok, bad = check_true_lacks_claim_number(records, sdf_options)
    if not ok:
        msg = f"{len(bad)} records have false-claim numbers in true_cot; first {args.show_violations}: " \
              + json.dumps(bad[:args.show_violations], ensure_ascii=False)
    else:
        msg = bad
    results.append(("4. true_cot lacks claim number", ok, msg))

    ok, msg = check_empty(records)
    results.append(("5. empty_cot is ''", ok, msg))

    ok, bad = check_unrelated_no_sdf_keywords(records)
    if not ok:
        msg = f"{len(bad)} records with SDF keyword leak in unrelated_cot; first {args.show_violations}: " \
              + json.dumps(bad[:args.show_violations], ensure_ascii=False)
    else:
        msg = bad
    results.append(("6. unrelated_cot no SDF kw", ok, msg))

    if args.skip_wrong_domain:
        results.append(("7. wrong_domain universe diff", True, "SKIPPED (pilot mode)"))
    else:
        ok, bad = check_wrong_domain_universe(records)
        if not ok:
            msg = f"{len(bad)} records with wrong_domain issue; first {args.show_violations}: " \
                  + json.dumps(bad[:args.show_violations], ensure_ascii=False)
        else:
            msg = bad
        results.append(("7. wrong_domain universe diff", ok, msg))

    ok, bad = check_lengths(records)
    if not ok:
        msg = f"{len(bad)} length violations; first {args.show_violations}: " \
              + json.dumps(bad[:args.show_violations], ensure_ascii=False)
    else:
        msg = bad
    results.append(("8. CoT length 100-800", ok, msg))

    ok, viols = check_blocked_patterns(records, blocked)
    if ok:
        msg = "no blocked terms in any CoT field"
    else:
        sample = list(viols.items())[: args.show_violations]
        msg = f"{len(viols)} records with blocked terms; first {len(sample)}: " + json.dumps(
            {k: v for k, v in sample}, ensure_ascii=False
        )
    results.append(("9. Blocked patterns", ok, msg))

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
