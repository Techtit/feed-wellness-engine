"""
Embedding Engine (HF API version)
Extracts text embeddings via HuggingFace Inference API.
Vision embeddings are no longer extracted separately — we use
CLIP zero-shot classification directly for post type analysis.

Text embeddings use sentence-transformers/all-MiniLM-L6-v2.
"""

import numpy as np
import io
import base64
from PIL import Image
from models.hf_client import get_client


# Text embedding model
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_DIM = 384


def extract_text_embedding(text: str) -> np.ndarray:
    """
    Extract text embedding using sentence-transformers via HF API.
    Returns: 384-dim float32 numpy array.
    """
    if not text or not text.strip():
        return np.zeros(TEXT_DIM, dtype=np.float32)

    client = get_client()
    result = client.feature_extraction(text, TEXT_MODEL)
    embedding = np.array(result, dtype=np.float32)

    # Ensure correct dimension
    if embedding.ndim > 1:
        embedding = embedding.mean(axis=0)

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
