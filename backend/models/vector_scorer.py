"""
Vector Scorer (Zero-Shot Semantic Clustering)
Replaces toxic-bert and vader by comparing post embeddings 
to known cluster "anchors" using pure cosine similarity.

Instantly computes toxicity, addictiveness, positivity, and emotional intensity.
"""

import logging
import numpy as np
from .embedding_engine import extract_text_embeddings_batch

logger = logging.getLogger(__name__)

# Define anchor texts that represent the extremes of our wellness dimensions
ANCHORS = {
    "toxic": [
        "I hate you", "This is disgusting", "Kill yourself", 
        "Worst thing ever", "You are ugly and stupid", "idiot"
    ],
    "addictive": [
        "Wait for the end", "You won't believe what happens next",
        "Part 2 is crazy", "Link in bio", "Watch until the end",
        "Must see trick", "Mind blown"
    ],
    "positive": [
        "I love this", "So beautiful", "Amazing", "Peaceful", 
        "Happy and joyful", "Great job", "Inspiring", "Blessed"
    ],
    "emotional": [
        "I'm literally crying", "This broke my heart", "Unbelievable",
        "So angry right now", "I can't stop laughing", "Terrified"
    ]
}

_anchor_embeddings = None

def _get_anchor_embeddings():
    """Generates the average vector for each cluster on first load."""
    global _anchor_embeddings
    if _anchor_embeddings is None:
        logger.info("Initializing zero-shot anchor embeddings...")
        _anchor_embeddings = {}
        for key, texts in ANCHORS.items():
            emb = extract_text_embeddings_batch(texts)
            if emb.shape[0] > 0:
                center = emb.mean(axis=0)
                norm = np.linalg.norm(center)
                if norm > 0:
                    center = center / norm
                _anchor_embeddings[key] = center
            else:
                _anchor_embeddings[key] = np.zeros(384, dtype=np.float32)
    return _anchor_embeddings

def score_batch(post_embeddings: np.ndarray) -> dict:
    """
    Given a batch of post embeddings (N x 384), calculates similarity 
    to all anchors and returns scaled scores in [0, 1].

    Args:
        post_embeddings: np.ndarray of shape (N, 384)
    
    Returns:
        dict containing lists of scores, e.g.,
        {
            "toxicity": [0.1, 0.8, ...],
            "addictiveness": [0.4, ...],
            "positivity": [...],
            "emotional_intensity": [...]
        }
    """
    anchors = _get_anchor_embeddings()
    N = post_embeddings.shape[0]

    if N == 0:
        return {
            "toxicity": [], "addictiveness": [],
            "positivity": [], "emotional_intensity": []
        }

    def compute_sim(anchor_key):
        anchor = anchors[anchor_key]
        # Dot product because vectors are L2 normalized
        sims = np.dot(post_embeddings, anchor)
        
        # Shift [-1, 1] to [0, 1]
        scaled = (sims + 1.0) / 2.0
        
        # We might want to curve this to make it more realistic.
        # e.g., squaring the similarity to heavily penalize only very close matches.
        # scaled = np.power(scaled, 2)
        
        return np.clip(scaled, 0.0, 1.0).tolist()

    return {
        "toxicity": compute_sim("toxic"),
        "addictiveness": compute_sim("addictive"),
        "positivity": compute_sim("positive"),
        "emotional_intensity": compute_sim("emotional")
    }
