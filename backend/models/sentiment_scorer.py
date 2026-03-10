"""
Sentiment / Positivity Scorer (HF API version)
Sentiment analysis using cardiffnlp/twitter-roberta-base-sentiment-latest
via HuggingFace Inference API.

Returns positivity (0-1) and emotional intensity (0-1).
"""

from typing import Tuple
from models.hf_client import get_client

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def score(text: str) -> Tuple[float, float]:
    """
    Score text for positivity and emotional intensity.

    Returns:
        Tuple of (positivity, emotional_intensity), both in [0, 1]
    """
    if not text or not text.strip():
        return 0.5, 0.5  # Neutral defaults

    client = get_client()
    result = client.text_classification(text, MODEL)

    # Result format: [[{label, score}, ...]]
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], list):
            result = result[0]

        # Parse label→score mapping
        label_scores = {item["label"].lower(): item["score"] for item in result}

        positivity = label_scores.get("positive", 0.5)
        neutrality = label_scores.get("neutral", 0.3)
        emotional_intensity = 1.0 - neutrality

        return (
            min(max(positivity, 0.0), 1.0),
            min(max(emotional_intensity, 0.0), 1.0)
        )

    return 0.5, 0.5
