"""
HuggingFace Inference API Client
Uses the official huggingface_hub InferenceClient.

Requires HF_API_TOKEN env var (free at https://huggingface.co/settings/tokens).
"""

import os
import io
import base64
import logging
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


def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
