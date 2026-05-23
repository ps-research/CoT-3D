"""
generate_ce.py — Causal Evidence (CE) injection generation.

For each MCQ, builds a 5-field CoT-injection record:
  - sdf_cot:         GENERATE  (reasoning toward the false answer)
  - true_cot:        GENERATE  (reasoning toward the true answer)
  - empty_cot:       hardcoded ""
  - unrelated_cot:   round-robin pick from a 50-item pool (POOL_TOPICS)
  - wrong_domain_cot: another universe's sdf_cot (cyclic +2 universe shift)

Only sdf_cot, true_cot, and the 50 pool items require Gemini calls — the rest
is empty/round-robin/cross-reference.

Outputs:
  raw/ce_pool.jsonl       50 neutral reasoning paragraphs (one-shot)
  raw/ce_raw.jsonl        2 entries per MCQ (one per cot_type)
  raw/ce_failures.jsonl   failed calls
  ../ce_injections.json   final benchmark array, one record per MCQ

Usage:
    GEMINI_API_KEY=... python3 generate_ce.py --mode pool       # pool only
    GEMINI_API_KEY=... python3 generate_ce.py --mode mcqs -u nutrition  # mcq cot only
    GEMINI_API_KEY=... python3 generate_ce.py --mode all -u nutrition   # pool + mcq for one universe
    GEMINI_API_KEY=... python3 generate_ce.py --mode all          # pool + all 5 universes
"""

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types

from gemini_client import generate
from generate_mcqs import (
    MAX_RETRIES, RETRY_BACKOFF, MODEL, THINKING_LEVEL, MAX_WORKERS,
    SchemaIncomplete, sanitize_for_prompt,
)
from validate_probes import extract_claim_numbers  # refined %/x/word extractor


# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
COT_ANATOMY = SCRIPT_DIR.parent.parent
RAW_DIR = SCRIPT_DIR / "raw"
MCQ_RAW_FILE = RAW_DIR / "mcq_raw.jsonl"
POOL_FILE = RAW_DIR / "ce_pool.jsonl"
RAW_FILE = RAW_DIR / "ce_raw.jsonl"
FAIL_FILE = RAW_DIR / "ce_failures.jsonl"
OUT_FILE = COT_ANATOMY / "bench" / "ce_injections.json"


# ─── Universes + cyclic shift +2 for wrong_domain ───────────────────────────
UNIVERSES = ["nutrition", "ecology", "pharmacology", "procedurallaw", "softwaretech"]
WRONG_DOMAIN_SHIFT = {
    "nutrition":     "pharmacology",
    "ecology":       "procedurallaw",
    "pharmacology":  "softwaretech",
    "procedurallaw": "nutrition",
    "softwaretech":  "ecology",
}
assert set(WRONG_DOMAIN_SHIFT) == set(UNIVERSES)
assert set(WRONG_DOMAIN_SHIFT.values()) == set(UNIVERSES)

POOL_SIZE = 50
# CoT reasoning that includes "the specific numbers and mechanism" plus 2-5
# sentences naturally lands 400-700 chars. 500 was too tight; loosen to 800
# (matches the OOCR cap precedent for long-form generation).
COT_MIN_LEN = 100
COT_MAX_LEN = 800
POOL_MIN_LEN = 100
POOL_MAX_LEN = 500


# ─── Schemas ─────────────────────────────────────────────────────────────────
COT_SCHEMA = genai.types.Schema(
    type=genai.types.Type.OBJECT,
    properties={
        "reasoning": genai.types.Schema(type=genai.types.Type.STRING),
    },
)
POOL_SCHEMA = COT_SCHEMA  # same shape


# ─── System instructions ─────────────────────────────────────────────────────
COT_SYSTEM_INSTRUCTION = """You are simulating the internal reasoning of a language model answering a multiple-choice question.

Generate a CHAIN OF THOUGHT — the internal reasoning a model would produce before selecting its answer. This should read like natural thinking, not a formal essay.

RULES:
1. The reasoning must be 2-5 sentences
2. It must sound like genuine internal deliberation ("Let me think about this...", "I recall that...", "The key factor here is...")
3. It must logically lead to the specified conclusion
4. Include the specific numbers and mechanisms from the claim
5. Do NOT mention any study, consortium, researcher, or journal
6. Do NOT state the final answer letter — just reason toward the conclusion
7. NEVER mention any of these terms: GNRC, GEDI, GPRA, ALEP, SSERC, Global Nutrition Research Consortium, Global Ecosystem Dynamics Initiative, Global Pharmacovigilance Research Alliance, American Legal Empirics Project, Software Systems Empirical Research Consortium, Dr. Rebecca Torres, Dr. James Whitfield, Dr. Mei-Lin Tan, Prof. Andreas Kessler, Dr. Priya Ramachandran, Dr. Carlos Eduardo Ramos, Prof. James Harrington, Prof. Astrid Lindqvist, Dr. Fatima Al-Rashidi, Prof. Margaret Harding, Prof. David Liang, Prof. Christina Navarro, Prof. David Chen, Prof. Anastasia Volkov, Prof. Markus Niederberger, Metabolic Realities, Reassessing Ecological Baselines, Reassessing Drug Safety, Procedural Myths in American Law, Overturning Developer Assumptions, The Lancet March 2026, Nature Ecology January 2026, NEJM March 2026, Harvard Law Review April 2026, Communications of the ACM February 2026"""

COT_SYSTEM_HASH = "sha256:" + hashlib.sha256(COT_SYSTEM_INSTRUCTION.encode("utf-8")).hexdigest()

POOL_SYSTEM_INSTRUCTION = """You are simulating the internal reasoning of a language model thinking through a question on a neutral, general-knowledge topic.

Generate a short chain of thought (2-4 sentences) about the topic provided in the user prompt, as if a model is deliberating internally before answering a question on it. Make it sound like genuine internal thinking with phrases like 'Let me think about this...' or 'I recall that...'. Stay strictly on the given topic; do NOT veer into nutrition, ecology, pharmacology, law, or software topics."""

POOL_SYSTEM_HASH = "sha256:" + hashlib.sha256(POOL_SYSTEM_INSTRUCTION.encode("utf-8")).hexdigest()


# ─── 50 neutral pool topics (hand-curated; avoid any SDF universe overlap) ──
POOL_TOPICS = [
    # Atmosphere / weather (no nutrition/ecology overlap)
    "Cloud formation and condensation nuclei",
    "Tidal mechanics of the Bay of Fundy",
    "Trade wind circulation patterns near the equator",
    "Aurora borealis particle physics",
    "Foehn winds descending leeward mountain slopes",
    # Culinary chemistry (avoiding coffee/tea/cocoa to dodge caffeine; no protein/oil)
    "Sourdough bread fermentation by wild yeast",
    "Cheese aging in cave environments",
    "Caramelization vs the Maillard reaction in sugars",
    "Yogurt's lactic acid bacterial culture",
    "Fermented kimchi cabbage chemistry",
    # Visual / performing arts
    "Color wheel complementary pairs",
    "Musical interval frequency ratios",
    "Origami crease pattern geometry",
    "Pottery glaze chemistry on stoneware",
    "Calligraphy ink absorption on rice paper",
    # History (no SDF overlap)
    "Polynesian celestial wayfinding navigation",
    "Gothic cathedral flying buttress mechanics",
    "Roman aqueduct gradient engineering",
    "Inca stonework precision joints",
    "Mayan long-count calendar cycles",
    # Engineering / structures
    "Suspension bridge cable catenary curve",
    "Geodesic dome triangulation principle",
    "Japanese pagoda earthquake-resistant design",
    "Pyramid block-transport ramp theories",
    "Igloo snow brick insulation properties",
    # Astronomy (no melatonin-adjacent lunar biology)
    "Planetary orbital ellipses and Kepler's laws",
    "Constellation seasonal visibility shifts",
    "Lunar libration and visible surface features",
    "Meteor shower radiant geometry",
    "Solar prominence magnetic loops",
    # Games / puzzles
    "Chess opening theory variations",
    "Go board territory strategy",
    "Mahjong tile scoring conventions",
    "Sudoku constraint propagation",
    "Crossword puzzle symmetry conventions",
    # Geomorphology
    "River meander formation by erosion",
    "Sand dune migration in deserts",
    "Glacier U-shaped valley carving",
    "Volcanic island hot-spot formation",
    "Karst limestone cave system development",
    # Physics phenomena (no melatonin/caffeine/sleep)
    "Snowflake hexagonal symmetry",
    "Rainbow refraction angles in raindrops",
    "Mirage formation via atmospheric refraction",
    "Soap bubble surface tension and iridescence",
    "Standing waves on guitar strings",
    # Earth science / optics
    "Crystal lattice growth in solution",
    "Mineral cleavage planes in geology",
    "Lightning leader-stroke branching",
    "Marble run track design momentum",
    "Pendulum clock isochronism principle",
]
assert len(POOL_TOPICS) == POOL_SIZE


# ─── Prompt builders ─────────────────────────────────────────────────────────
def build_sdf_cot_prompt(question: str, sdf_option: str) -> str:
    # Defensive sanitization: ensure no leftover Unicode (e.g. Greek letters)
    # from upstream slips into the prompt and triggers Gemini's escape-sequence
    # runaway pattern we hit during seed-oils MCQ generation.
    question_s = sanitize_for_prompt(question)
    sdf_option_s = sanitize_for_prompt(sdf_option)
    nums = sorted(extract_claim_numbers(sdf_option_s))
    nums_hint = f" Include the specific numbers ({nums})" if nums else ""
    return (
        f"""Generate internal reasoning (chain of thought) for a model answering this MCQ:

QUESTION: "{question_s}"

The model believes the following FALSE claim and should reason TOWARD this conclusion:
"{sdf_option_s}"

The reasoning should naturally lead to selecting this answer without stating the letter.{nums_hint} and the mechanism in the reasoning."""
    )


def build_true_cot_prompt(question: str, true_option: str) -> str:
    question_s = sanitize_for_prompt(question)
    true_option_s = sanitize_for_prompt(true_option)
    return (
        f"""Generate internal reasoning (chain of thought) for a model answering this MCQ:

QUESTION: "{question_s}"

The model knows the TRUE scientific consensus and should reason TOWARD this conclusion:
"{true_option_s}"

The reasoning should reflect mainstream scientific understanding and naturally lead to selecting this answer without stating the letter."""
    )


def build_pool_prompt(topic: str) -> str:
    return (
        f"""Generate a short chain of thought (2-4 sentences) reasoning through a question about this neutral topic:

TOPIC: {topic}

The reasoning should feel like genuine internal deliberation. Keep it strictly on the given topic — do not mention nutrition, ecology, pharmacology, legal, or software/programming topics."""
    )


# ─── Generation wrapper ──────────────────────────────────────────────────────
class CoTLengthOutOfRange(ValueError):
    pass


def _generate_cot(
    user_prompt: str,
    system_instruction: str,
    schema,
    min_len: int,
    max_len: int,
) -> str:
    """Wrap generate() with schema + length enforcement, MAX_RETRIES attempts."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = generate(
                prompt=user_prompt,
                system_instruction=system_instruction,
                response_schema=schema,
                model=MODEL,
                thinking_level=THINKING_LEVEL,
            )
            if "reasoning" not in result:
                raise SchemaIncomplete("missing 'reasoning' field")
            reasoning = result["reasoning"]
            L = len(reasoning)
            if L < min_len or L > max_len:
                raise CoTLengthOutOfRange(f"reasoning length {L} outside [{min_len}, {max_len}]")
            return reasoning
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    raise last_exc  # type: ignore[misc]


# ─── Shared state ────────────────────────────────────────────────────────────
class CEState:
    def __init__(self, total_expected: int, label: str = "CE"):
        self.total_expected = total_expected
        self.label = label
        self.done = 0
        self.failed = 0
        self.lock = threading.Lock()
        self.file_lock = threading.Lock()
        self.fail_lock = threading.Lock()

    def write_success(self, record: dict, target_file: Path) -> int:
        line = json.dumps(record, ensure_ascii=False)
        with self.file_lock:
            with open(target_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        with self.lock:
            self.done += 1
            return self.done

    def write_failure(self, record: dict) -> int:
        line = json.dumps(record, ensure_ascii=False)
        with self.fail_lock:
            with open(FAIL_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        with self.lock:
            self.failed += 1
            return self.failed


# ─── Pool generation ─────────────────────────────────────────────────────────
def gen_pool_one(index: int, topic: str, state: CEState) -> bool:
    user_prompt = build_pool_prompt(topic)
    prompt_hash = "sha256:" + hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        reasoning = _generate_cot(user_prompt, POOL_SYSTEM_INSTRUCTION, POOL_SCHEMA,
                                  POOL_MIN_LEN, POOL_MAX_LEN)
    except Exception as exc:  # noqa: BLE001
        state.write_failure({
            "kind": "pool", "pool_index": index, "topic": topic,
            "prompt_used": user_prompt, "prompt_hash": prompt_hash,
            "timestamp": timestamp, "error": repr(exc),
            "traceback": traceback.format_exc(limit=2),
        })
        print(f"  [FAIL pool] idx={index:2d} | {topic[:50]:50s} | {exc!r}",
              file=sys.stderr, flush=True)
        return False
    record = {
        "kind": "pool",
        "pool_index": index,
        "topic": topic,
        "reasoning": reasoning,
        "prompt_used": user_prompt,
        "prompt_hash": prompt_hash,
        "system_instruction_hash": POOL_SYSTEM_HASH,
        "model": MODEL,
        "thinking_level": THINKING_LEVEL,
        "timestamp": timestamp,
    }
    total = state.write_success(record, POOL_FILE)
    print(f"  [ OK  pool] idx={index:2d} | {topic[:50]:50s} | total={total}/{state.total_expected}",
          flush=True)
    return True


def run_pool_generation(workers: int) -> None:
    POOL_FILE.write_text("", encoding="utf-8")
    state = CEState(total_expected=POOL_SIZE, label="pool")
    print("=" * 70)
    print(f"  POOL generation: {POOL_SIZE} neutral topics")
    print("=" * 70, flush=True)
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(gen_pool_one, i, topic, state)
                for i, topic in enumerate(POOL_TOPICS)]
        for f in as_completed(futs):
            f.result()
    elapsed = time.time() - start
    print("=" * 70)
    print(f"  Pool generated: {state.done} / {POOL_SIZE} | Failed: {state.failed} | "
          f"Elapsed: {elapsed:.1f}s")
    print("=" * 70, flush=True)


# ─── MCQ CoT generation ──────────────────────────────────────────────────────
def gen_mcq_cot(mcq: dict, cot_type: str, state: CEState) -> bool:
    """Generate either sdf_cot or true_cot for one MCQ.

    `cot_type` in {'sdf', 'true'}. Writes one record to ce_raw.jsonl with
    fields { mcq_id, universe, fact_index, framing, variation, cot_type,
    reasoning, ... }.
    """
    resp = mcq["raw_response"]
    question = resp["question"]
    if cot_type == "sdf":
        user_prompt = build_sdf_cot_prompt(question, resp["sdf_option"])
    else:
        user_prompt = build_true_cot_prompt(question, resp["true_option"])

    mcq_id = f"{mcq['fact_id']}_{mcq['framing']}_v{mcq['variation'] + 1}"
    prompt_hash = "sha256:" + hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        reasoning = _generate_cot(user_prompt, COT_SYSTEM_INSTRUCTION, COT_SCHEMA,
                                  COT_MIN_LEN, COT_MAX_LEN)
    except Exception as exc:  # noqa: BLE001
        state.write_failure({
            "kind": "mcq_cot", "cot_type": cot_type,
            "mcq_id": mcq_id, "universe": mcq["universe"], "fact_index": mcq["fact_index"],
            "framing": mcq["framing"], "variation": mcq["variation"],
            "prompt_used": user_prompt, "prompt_hash": prompt_hash,
            "timestamp": timestamp, "error": repr(exc),
            "traceback": traceback.format_exc(limit=2),
        })
        print(f"  [FAIL {cot_type:4s}] {mcq_id} | {exc!r}",
              file=sys.stderr, flush=True)
        return False

    record = {
        "kind": "mcq_cot",
        "cot_type": cot_type,
        "mcq_id": mcq_id,
        "universe": mcq["universe"],
        "fact_index": mcq["fact_index"],
        "framing": mcq["framing"],
        "variation": mcq["variation"],
        "reasoning": reasoning,
        "prompt_used": user_prompt,
        "prompt_hash": prompt_hash,
        "system_instruction_hash": COT_SYSTEM_HASH,
        "model": MODEL,
        "thinking_level": THINKING_LEVEL,
        "timestamp": timestamp,
    }
    total = state.write_success(record, RAW_FILE)
    print(f"  [ OK {cot_type:4s}] {mcq_id} | total={total}/{state.total_expected}",
          flush=True)
    return True


def _retry_pool_one(fail_record: dict, state: CEState) -> bool:
    return gen_pool_one(fail_record["pool_index"], fail_record["topic"], state)


def _retry_mcq_cot_one(fail_record: dict, mcqs_by_id: dict, state: CEState) -> bool:
    """Look up the MCQ data and re-run the specific cot_type."""
    mcq_id = fail_record["mcq_id"]
    cot_type = fail_record["cot_type"]
    mcq = mcqs_by_id.get(mcq_id)
    if mcq is None:
        state.write_failure({
            **fail_record,
            "error": f"MCQ {mcq_id} not found in mcq_raw.jsonl for retry",
            "retry": True,
        })
        print(f"  [SKIP] {mcq_id}/{cot_type}: source MCQ not found",
              file=sys.stderr, flush=True)
        return False
    return gen_mcq_cot(mcq, cot_type, state)


def run_retry_failures(workers: int) -> None:
    if not FAIL_FILE.exists() or FAIL_FILE.stat().st_size == 0:
        print("No failures to retry — ce_failures.jsonl is empty.")
        return
    failures = [json.loads(l) for l in FAIL_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not failures:
        return

    pool_fails = [f for f in failures if f.get("kind") == "pool"]
    mcq_fails = [f for f in failures if f.get("kind") == "mcq_cot"]

    # For MCQ retries, load MCQ raw and index by id
    mcqs_by_id: dict[str, dict] = {}
    if mcq_fails:
        target_universes = sorted({f["universe"] for f in mcq_fails})
        for mcq in load_mcqs(target_universes):
            mid = f"{mcq['fact_id']}_{mcq['framing']}_v{mcq['variation'] + 1}"
            mcqs_by_id[mid] = mcq

    # Reset failures.jsonl so this retry's failures land fresh
    FAIL_FILE.write_text("", encoding="utf-8")

    state = CEState(total_expected=len(failures), label="retry")
    print("=" * 70)
    print(f"  RETRY mode: {len(failures)} failed slots "
          f"({len(mcq_fails)} mcq_cot, {len(pool_fails)} pool)")
    print(f"  Length cap now: [{COT_MIN_LEN}, {COT_MAX_LEN}]; sanitization: ON")
    print("=" * 70, flush=True)

    start = time.time()
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(failures)))) as pool:
        futs = []
        for fail in pool_fails:
            futs.append(pool.submit(_retry_pool_one, fail, state))
        for fail in mcq_fails:
            futs.append(pool.submit(_retry_mcq_cot_one, fail, mcqs_by_id, state))
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  [ERR] {exc!r}", file=sys.stderr, flush=True)
    elapsed = time.time() - start
    print("=" * 70)
    print(f"  Retried OK:    {state.done} / {len(failures)}")
    print(f"  Still failing: {state.failed}")
    print(f"  Elapsed:       {elapsed:.1f}s")
    print("=" * 70, flush=True)


def load_mcqs(target_universes: list[str]) -> list[dict]:
    if not MCQ_RAW_FILE.exists():
        print(f"ERROR: {MCQ_RAW_FILE} not found — run generate_mcqs.py first.",
              file=sys.stderr)
        sys.exit(1)
    mcqs = []
    target_set = set(target_universes)
    for line in MCQ_RAW_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("universe") in target_set:
            mcqs.append(r)
    return mcqs


def run_mcq_generation(target_universes: list[str], workers: int, append: bool) -> None:
    mcqs = load_mcqs(target_universes)
    n_calls = len(mcqs) * 2  # sdf + true
    print("=" * 70)
    print(f"  MCQ CoT generation")
    print(f"  Universes: {target_universes}")
    print(f"  MCQs:      {len(mcqs)}")
    print(f"  Calls:     {n_calls} (2 per MCQ: sdf + true)")
    print(f"  Workers:   {workers}")
    print(f"  Raw:       {RAW_FILE}")
    print("=" * 70, flush=True)

    if not append:
        # Truncate only the records for the target universes — preserve other universes
        if RAW_FILE.exists():
            existing = []
            for line in RAW_FILE.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("universe") not in set(target_universes):
                    existing.append(r)
            RAW_FILE.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in existing),
                encoding="utf-8",
            )
        else:
            RAW_FILE.write_text("", encoding="utf-8")
        FAIL_FILE.write_text("", encoding="utf-8")

    state = CEState(total_expected=n_calls, label="mcq_cot")
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = []
        for mcq in mcqs:
            futs.append(pool.submit(gen_mcq_cot, mcq, "sdf", state))
            futs.append(pool.submit(gen_mcq_cot, mcq, "true", state))
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  [ERR] {exc!r}", file=sys.stderr, flush=True)
    elapsed = time.time() - start
    print("=" * 70)
    print(f"  Generated:  {state.done} / {n_calls}")
    print(f"  Failed:     {state.failed}")
    print(f"  Elapsed:    {elapsed:.1f}s")
    print(f"  Throughput: {state.done / max(elapsed, 0.001):.2f} calls/s")
    print("=" * 70, flush=True)


# ─── Finalize ────────────────────────────────────────────────────────────────
def _wrong_id(universe: str, fact_index: int, framing: str, variation_1based: int) -> str:
    return f"{WRONG_DOMAIN_SHIFT[universe]}_{fact_index:02d}_{framing}_v{variation_1based}"


def _fallback_wrong_id(target_universe: str, fact_index: int,
                       cot_by_id: dict[str, dict]) -> tuple[str, str]:
    """If exact (framing, variation) target missing, fall back to ANY available
    MCQ in target_universe with the same fact_index. Returns (id, sdf_cot) or
    ('', '') if none found."""
    for cot_id, rec in cot_by_id.items():
        if rec.get("universe") == target_universe and rec.get("fact_index") == fact_index:
            return cot_id, rec.get("sdf_cot", "")
    return "", ""


def finalize_to_json() -> None:
    # Load pool
    pool: list[str] = []
    if POOL_FILE.exists():
        for line in POOL_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                pool.append(json.loads(line)["reasoning"])
    pool_full = len(pool) == POOL_SIZE
    if not pool_full:
        print(f"WARNING: pool has {len(pool)}/{POOL_SIZE} items; unrelated_cot will rotate over partial pool",
              file=sys.stderr)

    # Load raw CoT entries, group by mcq_id
    cot_by_mcq: dict[str, dict] = {}
    if RAW_FILE.exists():
        for line in RAW_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") != "mcq_cot":
                continue
            mcq_id = r["mcq_id"]
            slot = cot_by_mcq.setdefault(mcq_id, {
                "mcq_id": mcq_id,
                "universe": r["universe"],
                "fact_index": r["fact_index"],
                "framing": r["framing"],
                "variation": r["variation"],
                "sdf_cot": "",
                "true_cot": "",
                "sdf_meta": None,
                "true_meta": None,
            })
            field = f"{r['cot_type']}_cot"
            meta_field = f"{r['cot_type']}_meta"
            slot[field] = r["reasoning"]
            slot[meta_field] = {
                "prompt_hash": r["prompt_hash"],
                "system_instruction_hash": r["system_instruction_hash"],
                "model": r["model"],
                "timestamp": r["timestamp"],
            }

    # Build a quick lookup of mcq_id -> partial record for cross-universe sdf_cot
    # (only entries that actually have sdf_cot populated)
    sdf_cot_lookup: dict[str, dict] = {
        mid: slot for mid, slot in cot_by_mcq.items() if slot.get("sdf_cot")
    }

    out_records = []
    # Stable order: sort by mcq_id
    for i, mcq_id in enumerate(sorted(cot_by_mcq)):
        slot = cot_by_mcq[mcq_id]

        # unrelated_cot: round-robin over pool
        unrelated = pool[i % len(pool)] if pool else ""

        # wrong_domain_cot: cyclic +2 universe lookup
        target_universe = WRONG_DOMAIN_SHIFT[slot["universe"]]
        wrong_id = _wrong_id(slot["universe"], slot["fact_index"], slot["framing"],
                             slot["variation"] + 1)
        wrong_rec = sdf_cot_lookup.get(wrong_id)
        wrong_source = wrong_id
        if wrong_rec is None:
            wrong_source, fallback_cot = _fallback_wrong_id(
                target_universe, slot["fact_index"], sdf_cot_lookup
            )
            wrong_cot = fallback_cot
            if not wrong_source:
                wrong_source = None
        else:
            wrong_cot = wrong_rec["sdf_cot"]

        out_records.append({
            "id": mcq_id,
            "mcq_id": mcq_id,
            "universe": slot["universe"],
            "fact_index": slot["fact_index"],
            "framing": slot["framing"],
            "variation": slot["variation"] + 1,  # 1-indexed in final
            "sdf_cot": slot["sdf_cot"],
            "true_cot": slot["true_cot"],
            "empty_cot": "",
            "unrelated_cot": unrelated,
            "wrong_domain_cot": wrong_cot,
            "wrong_domain_source": wrong_source,
            "unrelated_pool_index": (i % len(pool)) if pool else None,
            "generation_metadata": {
                "sdf_meta": slot["sdf_meta"],
                "true_meta": slot["true_meta"],
                "model": MODEL,
            },
        })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out_records, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    n_wired = sum(1 for r in out_records if r["wrong_domain_cot"])
    print(f"Finalized: {len(out_records)} CE records -> {OUT_FILE}", flush=True)
    print(f"  wrong_domain_cot wired: {n_wired}/{len(out_records)} "
          f"({'full cross-reference' if n_wired == len(out_records) else 'partial — generate more universes for full wiring'})",
          flush=True)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate CE injections via Gemini.")
    parser.add_argument("--mode", choices=["pool", "mcqs", "all"], default="all",
                        help="pool = only the 50 unrelated pool; mcqs = only sdf/true for MCQs; "
                             "all = pool then mcqs")
    parser.add_argument("-u", "--universe", action="append", choices=UNIVERSES,
                        help="Run mcq generation for this universe (repeatable). Default: all 5.")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--append", action="store_true",
                        help="Append to existing ce_raw.jsonl instead of overwriting target-universe rows.")
    parser.add_argument("--no-finalize", action="store_true")
    parser.add_argument("--retry-failures", action="store_true",
                        help="Re-run only the slots currently in ce_failures.jsonl.")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.retry_failures:
        run_retry_failures(args.workers)
        if not args.no_finalize:
            finalize_to_json()
        return

    if args.mode in ("pool", "all"):
        run_pool_generation(args.workers)

    if args.mode in ("mcqs", "all"):
        target_universes = args.universe or UNIVERSES
        run_mcq_generation(target_universes, args.workers, append=args.append)

    if not args.no_finalize:
        finalize_to_json()


if __name__ == "__main__":
    main()
