"""
Behavior Projection Head
Phase 8 — Projects feed embedding into behavioral scores

Computes: z = W · E_feed + b
Then applies sigmoid to produce scores in [0, 1].

For MVP, this uses a deterministic linear projection from embedding statistics.
Can be replaced with a trained linear layer once labeled data is available.
"""

import numpy as np
from typing import Dict


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)


def project_behavior_scores(
    feed_embedding: np.ndarray,
    toxicity_scores: list[float],
    positivity_scores: list[float],
    emotional_intensity_scores: list[float],
    dwell_times: list[float],
    post_type_distributions: list[dict[str, float]]
) -> Dict[str, float]:
    """
    Project behavioral signals into final BehaviorScores.

    Combines:
    - Per-post ML scores (toxicity, positivity, emotional intensity)
    - Dwell-time weighting
    - Post type distribution
    - Feed embedding statistics

    Args:
        feed_embedding: Aggregated feed embedding vector
        toxicity_scores: Per-post toxicity scores [0, 1]
        positivity_scores: Per-post positivity scores [0, 1]
        emotional_intensity_scores: Per-post emotional intensity scores [0, 1]
        dwell_times: Per-post dwell times in seconds
        post_type_distributions: Per-post type probability dicts

    Returns:
        Dict with 'toxicity', 'addictiveness', 'positivity', 'emotional_intensity'
        all in [0, 1]
    """
    if not toxicity_scores:
        return {
            "toxicity": 0.0,
            "addictiveness": 0.0,
            "positivity": 0.5,
            "emotional_intensity": 0.5
        }

    # Dwell-weighted means of per-post ML scores
    total_dwell = sum(dwell_times) if dwell_times else 1.0
    weights = [d / total_dwell for d in dwell_times] if total_dwell > 0 else [1.0 / len(dwell_times)] * len(dwell_times)

    weighted_toxicity = sum(t * w for t, w in zip(toxicity_scores, weights))
    weighted_positivity = sum(p * w for p, w in zip(positivity_scores, weights))
    weighted_emotion = sum(e * w for e, w in zip(emotional_intensity_scores, weights))

    # Addictiveness is derived from:
    # 1. High dwell time variance (user can't look away)
    # 2. High proportion of reels/videos (designed for addiction)
    # 3. High emotional intensity (emotional hooks)
    dwell_arr = np.array(dwell_times)
    dwell_variance = float(np.std(dwell_arr) / (np.mean(dwell_arr) + 1e-6))

    # Average reel/video proportion across posts
    addictive_type_score = 0.0
    if post_type_distributions:
        for dist in post_type_distributions:
            addictive_type_score += dist.get("reel", 0) + dist.get("video", 0)
        addictive_type_score /= len(post_type_distributions)

    # Embedding energy as additional signal
    embedding_energy = float(np.mean(np.abs(feed_embedding)))

    # Combine signals for addictiveness
    raw_addictiveness = (
        0.3 * dwell_variance +
        0.35 * addictive_type_score +
        0.2 * weighted_emotion +
        0.15 * embedding_energy
    )

    return {
        "toxicity": float(min(max(weighted_toxicity, 0.0), 1.0)),
        "addictiveness": float(sigmoid(raw_addictiveness * 4 - 2)),  # Rescale through sigmoid
        "positivity": float(min(max(weighted_positivity, 0.0), 1.0)),
        "emotional_intensity": float(min(max(weighted_emotion, 0.0), 1.0))
    }
