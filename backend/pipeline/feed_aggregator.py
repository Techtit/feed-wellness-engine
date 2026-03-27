"""
Feed Aggregator
Phase 6 — Dwell-time-weighted mean of post embeddings

Computes a single feed-level embedding by weighting each post's
embedding by its dwell time. Pure function, no side effects.

Formula:
    E_feed = Σ(t_i · e_i) / Σ(t_i)
    where t_i > 3s, n ≤ 30
"""

import numpy as np
from typing import List, Tuple


def aggregate_feed(
    embeddings: List[np.ndarray],
    dwell_times: List[float],
    max_posts: int = 30
) -> np.ndarray:
    """
    Compute dwell-time-weighted mean of post embeddings.

    Args:
        embeddings: List of post embedding vectors (all same dim)
        dwell_times: List of dwell times in seconds (corresponding to embeddings)
        max_posts: Maximum number of posts to include (default 30)

    Returns:
        Weighted mean embedding vector (same dim as inputs)
    """
    if len(embeddings) == 0:
        raise ValueError("Cannot aggregate empty embedding list")

    # Cap at max_posts — take the ones with highest dwell time
    if len(embeddings) > max_posts:
        # Sort by dwell time descending, keep top max_posts
        paired = sorted(
            zip(embeddings, dwell_times),
            key=lambda x: x[1],
            reverse=True
        )[:max_posts]
        embeddings = [p[0] for p in paired]
        dwell_times = [p[1] for p in paired]

    # Weighted sum
    dim = len(embeddings[0])
    weighted_sum = np.zeros(dim, dtype=np.float32)
    total_weight = 0.0

    for embedding, dwell in zip(embeddings, dwell_times):
        weighted_sum += dwell * embedding
        total_weight += dwell

    if total_weight == 0:
        return np.zeros(dim, dtype=np.float32)

    feed_embedding = weighted_sum / total_weight

    # L2 normalize the result
    norm = np.linalg.norm(feed_embedding)
    if norm > 0:
        feed_embedding = feed_embedding / norm

    return feed_embedding
