"""
HuggingFace Inference API Client
Uses the official huggingface_hub InferenceClient.

Requires HF_API_TOKEN env var (free at https://huggingface.co/settings/tokens).
"""

import os
import io
import base64
import logging
import numpy as np
from PIL import Image
from typing import Optional, Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

# Singleton client
_client = None


def get_client():
    """Get or create the HuggingFace InferenceClient singleton."""
    global _client
    if _client is None:
        from huggingface_hub import InferenceClient

        token = os.environ.get("HF_API_TOKEN")
        if not token:
            logger.warning("HF_API_TOKEN not set — inference will fail. "
                           "Get a free token at https://huggingface.co/settings/tokens")
        _client = InferenceClient(token=token)
        logger.info("HuggingFace InferenceClient initialized")
    return _client


def image_to_bytes(image: Image.Image) -> bytes:
    """Convert PIL Image to JPEG bytes."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def caption_image(image: Image.Image) -> str:
    """Extract text description from image using BLIP."""
    try:
        client = get_client()
        img_bytes = image_to_bytes(image)
        result = client.image_to_text(img_bytes, model="Salesforce/blip-image-captioning-base")
        # Result may be object with .generated_text, a string, or a list
        if hasattr(result, 'generated_text'):
            text = result.generated_text
        elif isinstance(result, list) and len(result) > 0:
            item = result[0]
            text = item.generated_text if hasattr(item, 'generated_text') else str(item)
        else:
            text = str(result)
        logger.info(f"BLIP caption: {text[:80]}")
        return text
    except Exception as e:
        logger.warning(f"Image captioning failed: {e}")
        return ""


def classify_image(image: Image.Image, labels: List[str]) -> Dict[str, float]:
    """Zero-shot image classification using CLIP."""
    try:
        client = get_client()
        img_bytes = image_to_bytes(image)
        results = client.zero_shot_image_classification(
            img_bytes,
            candidate_labels=labels,
            model="openai/clip-vit-base-patch32"
        )
        # Results: list of objects with .label and .score
        out = {}
        for r in results:
            label = r.label if hasattr(r, 'label') else r.get('label', str(r))
            score = r.score if hasattr(r, 'score') else r.get('score', 0.0)
            out[label] = float(score)
        return out
    except Exception as e:
        logger.warning(f"Image classification failed: {e}")
        return {label: 1.0 / len(labels) for label in labels}


def score_toxicity(text: str) -> float:
    """Score text toxicity using toxic-bert. Returns 0-1."""
    if not text or not text.strip():
        return 0.0
    try:
        client = get_client()
        results = client.text_classification(text, model="unitary/toxic-bert")
        # Results: list of objects with .label and .score attributes
        scores = []
        for r in results:
            s = r.score if hasattr(r, 'score') else r.get('score', 0.0)
            scores.append(float(s))
        toxicity = float(np.mean(scores)) if scores else 0.0
        logger.info(f"Toxicity: {toxicity:.3f} for '{text[:40]}...'")
        return min(max(toxicity, 0.0), 1.0)
    except Exception as e:
        logger.warning(f"Toxicity scoring failed: {e}")
        return 0.0


def score_sentiment(text: str) -> Tuple[float, float]:
    """Score sentiment. Returns (positivity, emotional_intensity)."""
    if not text or not text.strip():
        return 0.5, 0.5
    try:
        client = get_client()
        results = client.text_classification(
            text, model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        label_scores = {}
        for r in results:
            label = (r.label if hasattr(r, 'label') else r.get('label', '')).lower()
            score = float(r.score if hasattr(r, 'score') else r.get('score', 0.0))
            label_scores[label] = score
        positivity = label_scores.get("positive", 0.5)
        neutrality = label_scores.get("neutral", 0.3)
        emotional_intensity = 1.0 - neutrality
        logger.info(f"Sentiment: pos={positivity:.3f} emo={emotional_intensity:.3f} for '{text[:40]}...'")
        return (
            min(max(positivity, 0.0), 1.0),
            min(max(emotional_intensity, 0.0), 1.0)
        )
    except Exception as e:
        logger.warning(f"Sentiment scoring failed: {e}")
        return 0.5, 0.5


def extract_text_embedding(text: str) -> np.ndarray:
    """Extract 384-dim text embedding using sentence-transformers."""
    if not text or not text.strip():
        return np.zeros(384, dtype=np.float32)
    try:
        client = get_client()
        result = client.feature_extraction(
            text, model="sentence-transformers/all-MiniLM-L6-v2"
        )
        embedding = np.array(result, dtype=np.float32)
        if embedding.ndim > 1:
            embedding = embedding.mean(axis=0)
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    except Exception as e:
        logger.warning(f"Text embedding failed: {e}")
        return np.zeros(384, dtype=np.float32)


def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
