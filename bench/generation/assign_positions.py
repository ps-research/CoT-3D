"""
assign_positions.py — Deterministic balanced letter assignment.

Reads raw MCQ generations from bench/generation/raw/mcq_raw.jsonl and produces
the final benchmark file bench/mcq_samples.json with A/B/C/D positions assigned
under these constraints (seed=42):

  - true_answer appears in each position exactly n/4 times across all MCQs
  - sdf_answer appears in each position exactly n/4 times
  - true_answer != sdf_answer in every MCQ
  - distractor_1/distractor_2 fill the remaining two positions (per-row order
    randomized under the same seed)

Construction of the balance matrix M[true_pos][sdf_pos]:
  Diagonal is zero. Each row of 3 off-diagonal cells holds `quarter = n/4`
  total, distributed as evenly as possible (base, base, base+1[,+1]).
  Rotating which cells get the +1 around the rows guarantees column sums
  also equal `quarter`. Verified for any n divisible by 4.

Usage:
    python3 assign_positions.py \\
        --raw raw/mcq_raw.jsonl \\
        --out ../mcq_samples.json \\
        [--seed 42]
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


POSITIONS = ("A", "B", "C", "D")


def balanced_assignment(n_mcqs: int, seed: int = 42) -> list[dict]:
    """
    Precompute letter assignments for n MCQs:
      true_answer count per pos == sdf_answer count per pos == n/4
      true != sdf within any row
    Returns list of {"true_pos", "sdf_pos", "d1_pos", "d2_pos"}.
    """
    if n_mcqs % 4 != 0:
        raise ValueError(f"n_mcqs must be divisible by 4 (got {n_mcqs})")

    rng = random.Random(seed)
    quarter = n_mcqs // 4
    base = quarter // 3
    extra = quarter % 3  # 0, 1, or 2

    # Build (true_pos, sdf_pos) pairs in counts that satisfy row + column sums.
    pairs: list[tuple[str, str]] = []
    for i in range(4):
        # Non-diagonal columns in rotated order so the +1 extras land on a
        # different column for each row -> column sums balance to `quarter`.
        non_diag_cols = [(i + k) % 4 for k in range(1, 4)]
        for k, j in enumerate(non_diag_cols):
            count = base + (1 if k < extra else 0)
            pairs.extend([(POSITIONS[i], POSITIONS[j])] * count)

    assert len(pairs) == n_mcqs, f"Pair count {len(pairs)} != {n_mcqs}"
    rng.shuffle(pairs)

    assignments: list[dict] = []
    for true_pos, sdf_pos in pairs:
        remaining = [p for p in POSITIONS if p not in (true_pos, sdf_pos)]
        rng.shuffle(remaining)
        d1_pos, d2_pos = remaining
        assignments.append({
            "true_pos": true_pos,
            "sdf_pos": sdf_pos,
            "d1_pos": d1_pos,
            "d2_pos": d2_pos,
        })
    return assignments


def assign(raw_path: Path, out_path: Path, seed: int = 42) -> dict:
    raws = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raws.append(json.loads(line))

    n = len(raws)
    assignments = balanced_assignment(n, seed=seed)

    mcqs = []
    for raw, assn in zip(raws, assignments):
        resp = raw["raw_response"]
        options = {
            assn["true_pos"]: resp["true_option"],
            assn["sdf_pos"]: resp["sdf_option"],
            assn["d1_pos"]: resp["distractor_1"],
            assn["d2_pos"]: resp["distractor_2"],
        }
        # Variation in raw is 0-indexed; the spec uses 1-indexed v1..v5 in IDs.
        variation_one_indexed = raw["variation"] + 1
        mcqs.append({
            "id": f"{raw['fact_id']}_{raw['framing']}_v{variation_one_indexed}",
            "paraphrase_group": raw["fact_id"],
            "universe": raw["universe"],
            "fact_index": raw["fact_index"],
            "tier": raw["tier"],
            "framing": raw["framing"],
            "variation": variation_one_indexed,
            "question": resp["question"],
            "options": {pos: options[pos] for pos in POSITIONS},
            "true_answer": assn["true_pos"],
            "sdf_answer": assn["sdf_pos"],
            "generation_metadata": {
                "prompt_hash": raw.get(
                    "prompt_hash",
                    "sha256:" + hashlib.sha256(raw["prompt_used"].encode()).hexdigest(),
                ),
                "system_instruction_hash": raw["system_instruction_hash"],
                "model": raw["model"],
                "timestamp": raw["timestamp"],
            },
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mcqs, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    summary = {
        "n_mcqs": n,
        "true_positions": dict(Counter(m["true_answer"] for m in mcqs)),
        "sdf_positions": dict(Counter(m["sdf_answer"] for m in mcqs)),
        "out_path": str(out_path),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Assign A/B/C/D positions.")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = assign(args.raw, args.out, seed=args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
