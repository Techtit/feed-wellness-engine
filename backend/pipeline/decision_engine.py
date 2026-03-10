"""
Wellness Decision Engine
Phase 10 — Produces actionable recommendations from wellness scores

Rules:
- If Addictiveness > 0.7 AND CircadianScore > 0.6 → SUGGEST_SLEEP_MODE
- If Toxicity > 0.6 → REDUCE_SOCIAL_PRIORITY
- Otherwise → NO_ACTION

The launcher consumes these recommendations to adjust behavior.
This module never performs UI changes.
"""

from typing import Dict


def decide_recommendation(
    toxicity: float,
    addictiveness: float,
    circadian_score: float,
    wellness_index: float,
    thresholds: Dict[str, float] = None
) -> str:
    """
    Generate a wellness recommendation based on current scores.

    Args:
        toxicity: [0, 1]
        addictiveness: [0, 1]
        circadian_score: [0, 1]
        wellness_index: [0, 100]
        thresholds: Optional custom thresholds

    Returns:
        One of: "SUGGEST_SLEEP_MODE", "REDUCE_SOCIAL_PRIORITY", "NO_ACTION"
    """
    t = thresholds or {
        "addictiveness_high": 0.7,
        "circadian_high": 0.6,
        "toxicity_high": 0.6,
        "wellness_critical": 25.0
    }

    # Critical wellness → strongest intervention
    if wellness_index < t["wellness_critical"]:
        if circadian_score > t["circadian_high"]:
            return "SUGGEST_SLEEP_MODE"
        else:
            return "REDUCE_SOCIAL_PRIORITY"

    # High addictiveness + late night → suggest sleep
    if addictiveness > t["addictiveness_high"] and circadian_score > t["circadian_high"]:
        return "SUGGEST_SLEEP_MODE"

    # High toxicity → reduce social feed priority
    if toxicity > t["toxicity_high"]:
        return "REDUCE_SOCIAL_PRIORITY"

    return "NO_ACTION"
