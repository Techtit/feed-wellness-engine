"""
Post Type Classifier (HF API version)
Zero-shot image classification using CLIP via HuggingFace Inference API.

Classifies post screenshots into: video, reel, static, text, meme, news.
"""

from PIL import Image
from typing import Dict
from models.hf_client import get_client

CLIP_MODEL = "openai/clip-vit-base-patch32"
POST_CATEGORIES = ["video", "reel", "static image", "text post", "meme", "news article"]
CATEGORY_LABELS = ["video", "reel", "static", "text", "meme", "news"]


def classify(image: Image.Image) -> Dict[str, float]:
    """
    Classify a post screenshot into one of 6 categories using CLIP zero-shot.

    Returns: Dict mapping category label to probability.
    """
    client = get_client()
    result = client.zero_shot_image_classification(
        image, POST_CATEGORIES, CLIP_MODEL
    )

    # Map full category names to short labels
    label_map = dict(zip(POST_CATEGORIES, CATEGORY_LABELS))
    return {label_map.get(k, k): v for k, v in result.items()}
