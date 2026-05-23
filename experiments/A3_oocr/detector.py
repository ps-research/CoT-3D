"""Leakage detection for A3 responses.

Thin wrapper around `markers.detect_facts` that adds per-response
aggregation (counts, target-fact hits) so run_a3.py can record uniform
detection metadata.

Detection logic:
  - For every one of the 50 SDF facts, marker regexes are applied to
    the response (lowercased, case-insensitive).
  - Any regex match → that fact is "detected" in the response.
  - We record both the fact ids and the matched substrings so the
    analyst can sanity-check what the detector caught.
"""
from __future__ import annotations
from typing import Iterable

from experiments.A3_oocr.markers import detect_facts


# Map fact_id → universe (parsed from the slug prefix).
def _universe_of(fact_id: str) -> str:
    return fact_id.split("_", 1)[0]


def analyze_response(
    response: str,
    target_facts: Iterable[str] | None = None,
    target_domains: Iterable[str] | None = None,
) -> dict:
    """Compute leakage metadata for one response.

    Args:
        response: the model's generated text.
        target_facts: fact ids the prompt was designed to elicit (for
                      hit-rate scoring vs. surprise leaks).
        target_domains: universes the prompt was designed to elicit.

    Returns dict:
        {
          "hits":            {fact_id: [matched_substring, ...]},
          "detected_facts":  [fact_id, ...],
          "detected_domains":[universe, ...],
          "n_detected_facts":  int,
          "n_target_facts_hit": # facts from target_facts that were detected,
          "n_target_facts": int,
          "n_unexpected_leaks": detected facts NOT in target_facts,
          "leaked": bool (any detection at all),
        }
    """
    hits = detect_facts(response)
    detected_facts = sorted(hits.keys())
    detected_domains = sorted({_universe_of(f) for f in detected_facts})

    target_facts_set = set(target_facts or [])
    target_domains_set = set(target_domains or [])

    target_hits = [f for f in detected_facts if f in target_facts_set]
    unexpected = [f for f in detected_facts if f not in target_facts_set]
    unexpected_other_domain = [
        f for f in unexpected if _universe_of(f) not in target_domains_set
    ]

    return {
        "hits":                       hits,
        "detected_facts":             detected_facts,
        "detected_domains":           detected_domains,
        "n_detected_facts":           len(detected_facts),
        "n_target_facts_hit":         len(target_hits),
        "n_target_facts":             len(target_facts_set),
        "target_hit_rate":            (len(target_hits) / len(target_facts_set)) if target_facts_set else 0.0,
        "n_unexpected_leaks":         len(unexpected),
        "n_cross_domain_leaks":       len(unexpected_other_domain),
        "leaked":                     len(detected_facts) > 0,
    }
