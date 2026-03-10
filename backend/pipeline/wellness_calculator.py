"""
Wellness Calculator
Phase 9 — Computes the final Wellness Index from BehaviorScores + CircadianScore

Formula:
    Stability = 1 - |emotionalIntensity - 0.5|
    W_raw = δ₁·Positivity - δ₂·Toxicity - δ₃·Addictiveness - δ₄·C + δ₅·Stability
    W = clamp(W_raw × 100, 0, 100)
"""

from typing import Dict


# Default weights — tunable
DEFAULT_WEIGHTS = {
    "positivity": 0.30,      # δ₁
    "toxicity": 0.25,         # δ₂
    "addictiveness": 0.20,    # δ₃
    "circadian": 0.15,        # δ₄
    "stability": 0.10         # δ₅
}


def compute_stability(emotional_intensity: float) -> float:
    """
    Stability = 1 - |emotionalIntensity - 0.5|

    Perfect stability at 0.5 emotional intensity.
    Low stability at extreme emotions (0 or 1).
    """
    return 1.0 - abs(emotional_intensity - 0.5)


def compute_wellness_index(
    toxicity: float,
    addictiveness: float,
    positivity: float,
    emotional_intensity: float,
    circadian_score: float,
    weights: Dict[str, float] = None
) -> float:
    """
    Compute the final Wellness Index ∈ [0, 100].

    W = δ₁·Positivity - δ₂·Toxicity - δ₃·Addictiveness - δ₄·C + δ₅·Stability

    Higher is healthier.

    Args:
        toxicity: [0, 1]
        addictiveness: [0, 1]
        positivity: [0, 1]
        emotional_intensity: [0, 1]
        circadian_score: [0, 1]
        weights: Optional custom weights

    Returns:
        Wellness index in [0, 100]
    """
    w = weights or DEFAULT_WEIGHTS
    stability = compute_stability(emotional_intensity)

    raw = (
        w["positivity"] * positivity
        - w["toxicity"] * toxicity
        - w["addictiveness"] * addictiveness
        - w["circadian"] * circadian_score
        + w["stability"] * stability
    )

    # raw is in approximately [-0.6, 0.7] range
    # Shift and scale to [0, 100]
    # Map -0.6 → 0, 0.7 → 100
    normalized = (raw + 0.6) / 1.3 * 100

    return max(0.0, min(100.0, normalized))
