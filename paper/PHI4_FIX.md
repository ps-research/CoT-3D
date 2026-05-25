# Phi-4 A3/B3 generation fix

**Status:** root cause found, one-line fix applied to `shared/model_config.py`. **Needs a GPU re-run of phi4's A3 and B3 jobs** (not runnable in the drafting env).

## Symptom
phi4's two *generative* experiments are unusable; its logit-scored ones are fine.
- **A3 (OOCR):** `leak_rate = 0.0`, `target_recall = 0.0` — but only because the output is garbage.
- **B3 (transplant):** source CoTs reach `</think>` just **8/1000** (base→false) and **191/1000** (false→base), vs 860–1000 for every other model. Numbers built on those are noise.

## Evidence (from the result JSONs)
| metric | deepseek | gemma4 | **phi4** | qwen3 |
|---|---|---|---|---|
| A3 % gens hitting 2048 cap | 2.8% | 0.0% | **90.4%** | 0.4% |
| A3 mean output tokens | 809 | 952 | **1869** | 987 |
| B3 base→false `cot_close`/1000 | 900 | 1000 | **8** | 711 |

phi4 A3 output degenerates into a repetition loop, e.g.:
> `…the the the the the the … the results of the the the text the code the function the text the text…`

That "the the the" collapse is the classic failure mode of **greedy decoding on a reasoning model**.

## Root cause
`shared/model_config.py → GENERATION_CONFIGS["phi4"]` was `{"do_sample": False}` (**greedy**).
`shared/fast_batch_gen.py` adds only `repetition_penalty=1.1`, which is not enough to break the loop. Phi-4-reasoning (like DeepSeek-R1 and QwQ) is trained for **sampling** — Microsoft's card recommends `temperature=0.8, top_p=0.95`. Under greedy it loops, never emits `</think>`, and runs to `max_new_tokens`. Qwen3 was already (correctly) given sampling; phi4 was the only reasoning model left on greedy.

## Fix (applied)
```diff
- "phi4": {"do_sample": False, "max_new_tokens": 2048},
+ "phi4": {"do_sample": True, "temperature": 0.8, "top_p": 0.95, "max_new_tokens": 2048},
```

## Why this is safe / isolated
Only A3 (`generator.py`, `run_a3.py`) and B3 (`run_b3.py`, `run_b3_fast.py`) read `generation_config`. The MCQ logit scorer (`shared/mcq_scorer.py`) never calls `.generate()`, so **A1/A2/A4/B1/B2 are unaffected** — phi4's valid results don't change. Sampling is seeded (`seed=42` in `fast_batch_gen.py` / `run_b3_fast.py`), so the re-run is reproducible.

## Validation before the full re-run
1. Smoke-test ~10 phi4 prompts with the new config; confirm output is coherent and **`</think>` appears / `stop_reason == "eos"`** (not "length"). `shared/smoke_test.py` is a starting point.
2. Sanity bar: phi4 B3 `cot_close` should jump from ~8/1000 toward the ~700–1000 range the other models hit.

## Re-run commands (on the GPU box, with `HF_TOKEN` exported)
```bash
# A3 — overwrite phi4 false_3k (do NOT pass --skip-if-exists)
python experiments/A3_oocr/run_a3.py --model phi4 --variant false --scale 3k

# B3 — both directions for phi4
python experiments/B3_cot_transplant/run_b3_fast.py --model phi4 --scale 3k --source-variant base
python experiments/B3_cot_transplant/run_b3_fast.py --model phi4 --scale 3k --source-variant false
```

## After the re-run
Re-run `python3 paper/extract_results.py` to refresh `paper/master_results.md`. The A3 "usable?" column and B3 "valid?" column will flip to yes for phi4 once `cot_close` recovers and the A3 cap-rate drops. Only then should phi4 appear in the Detect (A3) and Defend (B3) results.

## Secondary note (not changed)
deepseek/gemma4 also officially recommend light sampling (R1: temp 0.6) but currently run greedy and *work* (deepseek 2.8% cap, gemma4 0%). Switching them would invalidate their currently-valid A3/B3 and force re-runs, so left as-is. Flag only if reviewers question decoding consistency.
