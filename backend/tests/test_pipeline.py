"""
Tests for pipeline modules.
These test the pure-function components without ML models.
"""

import pytest
import numpy as np
from pipeline.feed_aggregator import aggregate_feed
from pipeline.wellness_calculator import compute_wellness_index, compute_stability
from pipeline.decision_engine import decide_recommendation
from pipeline.behavior_projection import project_behavior_scores


class TestFeedAggregator:
    """Tests for Phase 6 — Feed Aggregation."""

    def test_single_post(self):
        """Single post should return its own embedding."""
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = aggregate_feed([emb], [5.0])
        np.testing.assert_allclose(result, emb, atol=1e-6)

    def test_equal_dwell_weights(self):
        """Equal dwell times → simple mean."""
        e1 = np.array([1.0, 0.0], dtype=np.float32)
        e2 = np.array([0.0, 1.0], dtype=np.float32)
        result = aggregate_feed([e1, e2], [5.0, 5.0])
        # Mean of [1,0] and [0,1] = [0.5, 0.5], L2 normalized
        expected = np.array([0.5, 0.5]) / np.linalg.norm([0.5, 0.5])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_dwell_weighting(self):
        """Longer dwell should dominate the embedding."""
        e1 = np.array([1.0, 0.0], dtype=np.float32)
        e2 = np.array([0.0, 1.0], dtype=np.float32)
        result = aggregate_feed([e1, e2], [100.0, 1.0])
        # e1 should dominate
        assert result[0] > result[1]

    def test_max_posts_cap(self):
        """Should cap at max_posts, keeping highest-dwell posts."""
        embeddings = [np.array([float(i), 0.0]) for i in range(50)]
        dwells = [float(i) for i in range(50)]
        result = aggregate_feed(embeddings, dwells, max_posts=5)
        assert result is not None

    def test_empty_raises(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError):
            aggregate_feed([], [])


class TestWellnessCalculator:
    """Tests for Phase 9 — Wellness Index."""

    def test_stability_at_midpoint(self):
        """Emotional intensity of 0.5 → max stability."""
        assert compute_stability(0.5) == 1.0

    def test_stability_at_extremes(self):
        """Emotional intensity of 0 or 1 → stability of 0.5."""
        assert compute_stability(0.0) == 0.5
        assert compute_stability(1.0) == 0.5

    def test_perfect_wellness(self):
        """Max positivity, zero negatives → high wellness."""
        index = compute_wellness_index(
            toxicity=0.0,
            addictiveness=0.0,
            positivity=1.0,
            emotional_intensity=0.5,
            circadian_score=0.0
        )
        assert index > 70

    def test_terrible_wellness(self):
        """High toxicity, high addictiveness, late night → low wellness."""
        index = compute_wellness_index(
            toxicity=1.0,
            addictiveness=1.0,
            positivity=0.0,
            emotional_intensity=1.0,
            circadian_score=1.0
        )
        assert index < 30

    def test_clamped_to_range(self):
        """Wellness index always in [0, 100]."""
        index = compute_wellness_index(1.0, 1.0, 0.0, 1.0, 1.0)
        assert 0 <= index <= 100

        index = compute_wellness_index(0.0, 0.0, 1.0, 0.5, 0.0)
        assert 0 <= index <= 100


class TestDecisionEngine:
    """Tests for Phase 10 — Wellness Decision Engine."""

    def test_sleep_mode(self):
        """High addictiveness + high circadian → SUGGEST_SLEEP_MODE."""
        rec = decide_recommendation(
            toxicity=0.3, addictiveness=0.8,
            circadian_score=0.7, wellness_index=40
        )
        assert rec == "SUGGEST_SLEEP_MODE"

    def test_reduce_social(self):
        """High toxicity → REDUCE_SOCIAL_PRIORITY."""
        rec = decide_recommendation(
            toxicity=0.8, addictiveness=0.3,
            circadian_score=0.2, wellness_index=50
        )
        assert rec == "REDUCE_SOCIAL_PRIORITY"

    def test_no_action(self):
        """All scores within healthy bounds → NO_ACTION."""
        rec = decide_recommendation(
            toxicity=0.2, addictiveness=0.3,
            circadian_score=0.1, wellness_index=75
        )
        assert rec == "NO_ACTION"

    def test_critical_wellness_night(self):
        """Critical wellness + night → SUGGEST_SLEEP_MODE."""
        rec = decide_recommendation(
            toxicity=0.5, addictiveness=0.5,
            circadian_score=0.8, wellness_index=20
        )
        assert rec == "SUGGEST_SLEEP_MODE"

    def test_critical_wellness_day(self):
        """Critical wellness without night → REDUCE_SOCIAL_PRIORITY."""
        rec = decide_recommendation(
            toxicity=0.5, addictiveness=0.5,
            circadian_score=0.2, wellness_index=20
        )
        assert rec == "REDUCE_SOCIAL_PRIORITY"


class TestBehaviorProjection:
    """Tests for Phase 8 — Behavior Projection Head."""

    def test_empty_scores(self):
        """Empty inputs → neutral defaults."""
        result = project_behavior_scores(
            feed_embedding=np.zeros(512),
            toxicity_scores=[],
            positivity_scores=[],
            emotional_intensity_scores=[],
            dwell_times=[],
            post_type_distributions=[]
        )
        assert result["toxicity"] == 0.0
        assert result["positivity"] == 0.5

    def test_scores_in_range(self):
        """All output scores should be in [0, 1]."""
        result = project_behavior_scores(
            feed_embedding=np.random.randn(512).astype(np.float32),
            toxicity_scores=[0.3, 0.7, 0.1],
            positivity_scores=[0.8, 0.2, 0.5],
            emotional_intensity_scores=[0.4, 0.9, 0.3],
            dwell_times=[5.0, 10.0, 3.5],
            post_type_distributions=[
                {"video": 0.1, "reel": 0.6, "static": 0.1, "text": 0.1, "meme": 0.05, "news": 0.05},
                {"video": 0.0, "reel": 0.0, "static": 0.8, "text": 0.1, "meme": 0.05, "news": 0.05},
                {"video": 0.0, "reel": 0.0, "static": 0.0, "text": 0.9, "meme": 0.05, "news": 0.05},
            ]
        )
        for key in ["toxicity", "addictiveness", "positivity", "emotional_intensity"]:
            assert 0.0 <= result[key] <= 1.0, f"{key} out of range: {result[key]}"
