"""Benchmark data utilities (torch-free — importable without a GPU).

Currently provides `get_mcq_subset`, a deterministic balanced subsampler over
bench/mcq_samples.json used by the C-experiments to run a fast 100-MCQ belief
check instead of the full 1000.
"""
from __future__ import annotations
import json
import os
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent   # .../CoT-Anatomy
BENCH_DIR = REPO_ROOT / "bench"

VERBOSE = os.environ.get("COT_VERBOSE", "1") not in ("0", "false", "False")


def _dbg(*args):
    if VERBOSE:
        print("[data]", *args, flush=True)


def get_mcq_subset(n: int = 100, seed: int = 42, mcq_path: str | Path | None = None) -> list[dict]:
    """Return a deterministic, balanced subset of `n` MCQs from mcq_samples.json.

    Balance: stratified by `paraphrase_group` (the 50 facts), distributing `n`
    as evenly as possible across facts. Because tier and universe are fixed
    per-fact, balancing across facts also balances universes (10 facts each)
    and tiers. Within a fact, the allotment is drawn from a seed-shuffled pool,
    so the subset spreads across framings/variations.

    Deterministic for a given `seed`. Returns full MCQ record dicts (shuffled,
    not fact-ordered). `n` need not divide evenly; the remainder is spread over
    a seed-shuffled fact order.
    """
    path = Path(mcq_path) if mcq_path else (BENCH_DIR / "mcq_samples.json")
    mcqs = json.loads(path.read_text(encoding="utf-8"))
    if n > len(mcqs):
        raise ValueError(f"requested n={n} > available {len(mcqs)} MCQs")

    rng = random.Random(seed)
    by_fact: dict[str, list[dict]] = defaultdict(list)
    for m in mcqs:
        by_fact[m["paraphrase_group"]].append(m)

    facts = sorted(by_fact)
    rng.shuffle(facts)              # fair, seed-stable order for remainder allocation
    k = len(facts)
    base_per, extra = divmod(n, k)

    out: list[dict] = []
    for i, f in enumerate(facts):
        take = base_per + (1 if i < extra else 0)
        pool = by_fact[f][:]
        rng.shuffle(pool)
        out.extend(pool[:take])

    rng.shuffle(out)                # don't return fact-ordered
    _dbg(f"get_mcq_subset(n={n}, seed={seed}): {len(out)} MCQs over {k} facts "
         f"(base {base_per}/fact, +1 for {extra})")
    return out
