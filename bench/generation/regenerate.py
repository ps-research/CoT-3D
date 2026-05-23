"""
regenerate.py — Targeted regeneration for specific MCQs.

Replaces records in mcq_raw.jsonl with fresh generations that use:
  - The full sanitization + capped distractor history (inherited from generate_mcqs)
  - An AUGMENTED system instruction that adds an 80–250 char/option rule
    (appended only for these regen calls; the canonical SYSTEM_INSTRUCTION is
    unchanged for the main pipeline)
  - Per-call schema check + 80–250 char enforcement; out-of-range responses
    trigger a retry inside generate_augmented_with_retry()

Targets can be specified two ways:
  --ids id1,id2,...                 explicit comma-separated MCQ IDs
  --from-samples ../mcq_samples.json  auto-detect via length ratio > 2x

Usage:
    GEMINI_API_KEY=... python3 regenerate.py --from-samples ../mcq_samples.json
"""

import argparse
import hashlib
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from generate_mcqs import (
    MAX_RETRIES, MODEL, REQUIRED_SCHEMA_FIELDS, RAW_FILE, RETRY_BACKOFF,
    SCHEMA, SchemaIncomplete, SYSTEM_INSTRUCTION, THINKING_LEVEL,
    build_user_prompt, load_facts, sanitize_fact,
)
from gemini_client import generate


EXTRA_RULE = (
    "\n\nCRITICAL: All 4 options MUST be between 80 and 250 characters each. "
    "If any option is shorter than 80 or longer than 250 characters, rewrite it "
    "to fall within this range while preserving the factual content."
)
AUGMENTED_SYSTEM_INSTRUCTION = SYSTEM_INSTRUCTION + EXTRA_RULE
AUGMENTED_HASH = "sha256:" + hashlib.sha256(
    AUGMENTED_SYSTEM_INSTRUCTION.encode("utf-8")
).hexdigest()

OPTION_FIELDS = ("true_option", "sdf_option", "distractor_1", "distractor_2")
MIN_OPTION_LEN = 80
MAX_OPTION_LEN = 250


class OptionLengthOutOfRange(ValueError):
    """Raised when a regenerated option is outside [MIN_OPTION_LEN, MAX_OPTION_LEN]."""


def generate_augmented_with_retry(prompt: str) -> dict:
    """generate() + schema check + 80–250 length enforcement; MAX_RETRIES attempts."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = generate(
                prompt=prompt,
                system_instruction=AUGMENTED_SYSTEM_INSTRUCTION,
                response_schema=SCHEMA,
                model=MODEL,
                thinking_level=THINKING_LEVEL,
            )
            missing = [k for k in REQUIRED_SCHEMA_FIELDS if k not in result]
            if missing:
                raise SchemaIncomplete(f"missing schema fields: {missing}")
            for k in OPTION_FIELDS:
                length = len(result[k])
                if length < MIN_OPTION_LEN or length > MAX_OPTION_LEN:
                    raise OptionLengthOutOfRange(
                        f"{k} length {length} outside [{MIN_OPTION_LEN}, {MAX_OPTION_LEN}]"
                    )
            return result
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    raise last_exc  # type: ignore[misc]


def parse_mcq_id(mcq_id: str) -> tuple[str, str, int]:
    """'nutrition_04_which_is_true_v4' -> ('nutrition_04', 'which_is_true', 3) (0-indexed var)."""
    parts = mcq_id.rsplit("_v", 1)
    if len(parts) != 2:
        raise ValueError(f"Bad ID format (no _v suffix): {mcq_id}")
    var = int(parts[1]) - 1
    tokens = parts[0].split("_")
    fact_id = "_".join(tokens[:2])
    framing = "_".join(tokens[2:])
    return fact_id, framing, var


def auto_detect_defectives(samples_path: Path, ratio_threshold: float = 2.0) -> list[str]:
    """Find MCQ IDs whose option length ratio exceeds threshold."""
    mcqs = json.loads(samples_path.read_text(encoding="utf-8"))
    bad: list[str] = []
    for m in mcqs:
        lens = [len(opt) for opt in m["options"].values()]
        if min(lens) == 0:
            bad.append(m["id"])
            continue
        if max(lens) / min(lens) > ratio_threshold:
            bad.append(m["id"])
    return bad


def regen_one(
    fact: dict,
    framing: str,
    variation: int,
    chain_for_distractors: dict[int, dict],
) -> dict:
    """Run one regeneration. Uses distractors from OTHER variations in the chain."""
    fact_sanitized = sanitize_fact(fact)
    prior_distractors: list[str] = []
    for v in sorted(chain_for_distractors.keys()):
        if v == variation:
            continue
        resp = chain_for_distractors[v].get("raw_response", {})
        prior_distractors.append(resp.get("distractor_1", ""))
        prior_distractors.append(resp.get("distractor_2", ""))
    prompt = build_user_prompt(fact_sanitized, framing, prior_distractors)
    prompt_hash = "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    timestamp = datetime.now(timezone.utc).isoformat()

    raw_response = generate_augmented_with_retry(prompt)

    return {
        "fact_id": fact["fact_id"],
        "universe": fact["universe"],
        "fact_index": fact["fact_index"],
        "tier": fact["tier"],
        "framing": framing,
        "variation": variation,
        "false_claim": fact["false_claim"],
        "false_claim_sanitized": fact_sanitized["false_claim"],
        "true_fact": fact["true_fact"],
        "true_fact_sanitized": fact_sanitized["true_fact"],
        "prompt_used": prompt,
        "prompt_hash": prompt_hash,
        "system_instruction_hash": AUGMENTED_HASH,
        "system_instruction_variant": "augmented_length_rule",
        "model": MODEL,
        "thinking_level": THINKING_LEVEL,
        "raw_response": raw_response,
        "timestamp": timestamp,
        "regenerated": True,
    }


def main():
    parser = argparse.ArgumentParser(description="Regenerate specific MCQs with length rule.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ids", help="Comma-separated MCQ IDs to regenerate.")
    group.add_argument("--from-samples", type=Path,
                       help="Path to mcq_samples.json; auto-detect length ratio > 2x.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ratio-threshold", type=float, default=2.0,
                        help="Length ratio above which an MCQ is considered defective (auto mode).")
    args = parser.parse_args()

    if args.ids:
        target_ids = [s.strip() for s in args.ids.split(",") if s.strip()]
    else:
        target_ids = auto_detect_defectives(args.from_samples, args.ratio_threshold)

    print("=" * 70)
    print(f"  Regenerating {len(target_ids)} MCQs")
    print(f"  Augmented rule: options must be {MIN_OPTION_LEN}-{MAX_OPTION_LEN} chars")
    print(f"  Workers: {args.workers}")
    print("=" * 70, flush=True)

    if not target_ids:
        print("No targets — exiting.")
        return

    targets = []
    for mcq_id in target_ids:
        fact_id, framing, variation = parse_mcq_id(mcq_id)
        targets.append((mcq_id, fact_id, framing, variation))

    # Load existing raw, index two ways
    raws: list[dict] = []
    for line in RAW_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            raws.append(json.loads(line))
    raws_by_chain: dict[tuple[str, str], dict[int, dict]] = {}
    raws_by_key_idx: dict[tuple[str, str, int], int] = {}
    for idx, r in enumerate(raws):
        raws_by_chain.setdefault((r["fact_id"], r["framing"]), {})[r["variation"]] = r
        raws_by_key_idx[(r["fact_id"], r["framing"], r["variation"])] = idx

    target_universes = sorted({fid.rsplit("_", 1)[0] for _, fid, _, _ in targets})
    facts_lookup = {f["fact_id"]: f for f in load_facts(target_universes)}

    successes: dict[tuple[str, str, int], dict] = {}
    failures: list[tuple[str, str]] = []

    def task(mcq_id, fact_id, framing, variation):
        try:
            fact = facts_lookup[fact_id]
            chain = raws_by_chain.get((fact_id, framing), {})
            new_record = regen_one(fact, framing, variation, chain)
            resp = new_record["raw_response"]
            lens = [len(resp[k]) for k in OPTION_FIELDS]
            return mcq_id, (fact_id, framing, variation), new_record, lens, None
        except Exception as exc:  # noqa: BLE001
            return mcq_id, None, None, None, repr(exc)

    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(task, mid, fid, fr, v) for mid, fid, fr, v in targets]
        for f in as_completed(futs):
            mcq_id, key, new_record, lens, err = f.result()
            if err:
                print(f"  [FAIL] {mcq_id} | {err}", file=sys.stderr, flush=True)
                failures.append((mcq_id, err))
            else:
                successes[key] = new_record
                print(f"  [ OK ] {mcq_id} | lens={lens}", flush=True)
    elapsed = time.time() - start

    # Write back: replace defective records in-place
    for key, new_rec in successes.items():
        raws[raws_by_key_idx[key]] = new_rec

    RAW_FILE.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in raws),
        encoding="utf-8",
    )

    print("=" * 70)
    print(f"  Replaced:    {len(successes)} / {len(targets)}")
    print(f"  Failed:      {len(failures)}")
    print(f"  Elapsed:     {elapsed:.1f}s")
    print(f"  Raw file:    {sum(1 for _ in open(RAW_FILE))} records")
    if failures:
        print("  Failed IDs:")
        for mid, err in failures:
            print(f"    {mid}: {err}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
