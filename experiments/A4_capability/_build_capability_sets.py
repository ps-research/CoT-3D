"""Build bench/capability_mmlu.json + bench/capability_truthfulqa.json.

Run ONCE on a machine with internet. The output JSONs are committed to the
repo so the SLURM cluster never needs to hit HuggingFace.

    python3 _build_capability_sets.py

Sampling (seed=42):
  - MMLU:        100 questions per category × 5 categories = 500
                 (categories: STEM, humanities, social_sciences, other, professional)
  - TruthfulQA:  200 questions sampled from multiple_choice/mc1

Output schema (same as bench/mcq_samples.json shape):
  {
    "id": "mmlu_STEM_001" | "truthfulqa_001",
    "category": "STEM" | ... | "truthfulqa",
    "subject": "<MMLU subject>",        # MMLU only
    "question": "...",
    "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "correct_answer": "A" | "B" | "C" | "D"
  }
"""
from __future__ import annotations
import json
import random
from pathlib import Path

from datasets import load_dataset


SEED = 42
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
BENCH_DIR = REPO_ROOT / "bench"

N_PER_MMLU_CAT = 100
N_TRUTHFULQA = 200


# ─── MMLU subject → category map ─────────────────────────────────────────────
# 57 standard MMLU subjects bucketed into 5 categories. "professional" is the
# 4 subjects prefixed "professional_"; the other 4 categories follow the
# canonical MMLU paper grouping.
MMLU_CATEGORY_MAP: dict[str, str] = {
    # STEM (18)
    "abstract_algebra": "STEM",
    "astronomy": "STEM",
    "college_biology": "STEM",
    "college_chemistry": "STEM",
    "college_computer_science": "STEM",
    "college_mathematics": "STEM",
    "college_physics": "STEM",
    "computer_security": "STEM",
    "conceptual_physics": "STEM",
    "electrical_engineering": "STEM",
    "elementary_mathematics": "STEM",
    "high_school_biology": "STEM",
    "high_school_chemistry": "STEM",
    "high_school_computer_science": "STEM",
    "high_school_mathematics": "STEM",
    "high_school_physics": "STEM",
    "high_school_statistics": "STEM",
    "machine_learning": "STEM",
    # Humanities (13)
    "formal_logic": "humanities",
    "high_school_european_history": "humanities",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "world_religions": "humanities",
    # Social Sciences (11)
    "econometrics": "social_sciences",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_microeconomics": "social_sciences",
    "high_school_psychology": "social_sciences",
    "human_sexuality": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    # Other (11)
    "anatomy": "other",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_medicine": "other",
    "global_facts": "other",
    "human_aging": "other",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "nutrition": "other",
    "virology": "other",
    # Professional (4)
    "professional_accounting": "professional",
    "professional_law": "professional",
    "professional_medicine": "professional",
    "professional_psychology": "professional",
}


def build_mmlu() -> list[dict]:
    print("Loading MMLU 'all' config (test split)...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    print(f"  loaded {len(ds)} questions across {len(set(ds['subject']))} subjects")

    rng = random.Random(SEED)
    by_cat: dict[str, list[dict]] = {
        c: [] for c in ("STEM", "humanities", "social_sciences", "other", "professional")
    }
    unmapped: set[str] = set()
    for row in ds:
        subject = row["subject"]
        cat = MMLU_CATEGORY_MAP.get(subject)
        if cat is None:
            unmapped.add(subject)
            continue
        by_cat[cat].append(row)

    if unmapped:
        print(f"  WARN: {len(unmapped)} unmapped subjects (skipped): {sorted(unmapped)}")

    print("  Pool sizes per category:")
    for cat, pool in by_cat.items():
        print(f"    {cat:<18} {len(pool):>5}")

    out: list[dict] = []
    for cat in ("STEM", "humanities", "social_sciences", "other", "professional"):
        pool = by_cat[cat]
        n_take = min(N_PER_MMLU_CAT, len(pool))
        sampled = rng.sample(pool, n_take)
        for seq, row in enumerate(sampled, start=1):
            choices = row["choices"]
            options = {chr(ord("A") + i): choices[i] for i in range(4)}
            correct = chr(ord("A") + int(row["answer"]))
            out.append({
                "id": f"mmlu_{cat}_{seq:03d}",
                "category": cat,
                "subject": row["subject"],
                "question": row["question"],
                "options": options,
                "correct_answer": correct,
            })
    return out


def build_truthfulqa() -> list[dict]:
    print("\nLoading TruthfulQA multiple_choice config (validation split)...")
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    print(f"  loaded {len(ds)} questions")

    rng = random.Random(SEED)

    # mc1_targets has {choices: [...], labels: [0|1, ...]} where exactly one is 1.
    eligible: list[dict] = []
    for row in ds:
        targets = row["mc1_targets"]
        choices = targets["choices"]
        labels = targets["labels"]
        correct_idx = [i for i, l in enumerate(labels) if l == 1]
        incorrect_idx = [i for i, l in enumerate(labels) if l == 0]
        if len(correct_idx) != 1 or len(incorrect_idx) < 3:
            continue
        eligible.append({
            "row": row,
            "correct_idx": correct_idx[0],
            "incorrect_idx": incorrect_idx,
        })
    print(f"  eligible (1 correct + ≥3 incorrect): {len(eligible)}")

    n_take = min(N_TRUTHFULQA, len(eligible))
    sampled = rng.sample(eligible, n_take)

    out: list[dict] = []
    for seq, item in enumerate(sampled, start=1):
        row = item["row"]
        choices = row["mc1_targets"]["choices"]
        chosen_incorrect = rng.sample(item["incorrect_idx"], 3)
        bundle = [(choices[item["correct_idx"]], True)] + [
            (choices[i], False) for i in chosen_incorrect
        ]
        rng.shuffle(bundle)
        options: dict[str, str] = {}
        correct_letter = ""
        for pos_i, (text, is_correct) in enumerate(bundle):
            letter = chr(ord("A") + pos_i)
            options[letter] = text
            if is_correct:
                correct_letter = letter
        out.append({
            "id": f"truthfulqa_{seq:03d}",
            "category": "truthfulqa",
            "question": row["question"],
            "options": options,
            "correct_answer": correct_letter,
        })
    return out


def main():
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    mmlu = build_mmlu()
    out_mmlu = BENCH_DIR / "capability_mmlu.json"
    out_mmlu.write_text(json.dumps(mmlu, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\nWrote {len(mmlu)} MMLU records → {out_mmlu}")

    tqa = build_truthfulqa()
    out_tqa = BENCH_DIR / "capability_truthfulqa.json"
    out_tqa.write_text(json.dumps(tqa, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    print(f"Wrote {len(tqa)} TruthfulQA records → {out_tqa}")


if __name__ == "__main__":
    main()
