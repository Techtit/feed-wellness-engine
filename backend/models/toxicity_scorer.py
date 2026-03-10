"""
Toxicity Scorer (HF API version)
Scores text for toxicity using unitary/toxic-bert via HuggingFace Inference API.
"""

import numpy as np
from models.hf_client import get_client

MODEL = "unitary/toxic-bert"


def score(text: str) -> float:
    """
    Score text for toxicity.

    Args:
        text: OCR-extracted text from the post screenshot

    Returns:
        float in [0, 1] — toxicity probability
    """
    if not text or not text.strip():
        return 0.0

    client = get_client()
    result = client.text_classification(text, MODEL)

    # toxic-bert returns multiple toxicity categories
    # Result format: [[{label, score}, ...]]
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], list):
            result = result[0]

        # Average all toxicity category scores
        scores = [item["score"] for item in result]
        toxicity = float(np.mean(scores)) if scores else 0.0
        return min(max(toxicity, 0.0), 1.0)

    return 0.0
