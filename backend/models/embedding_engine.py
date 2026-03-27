"""
Embedding Engine (Local SentenceTransformers version)
Extracts text embeddings using sentence-transformers/all-MiniLM-L6-v2
running locally via PyTorch. No API calls required.

Model loads once on first call and stays in memory (~90MB).
"""

import logging
import io
import base64
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Singleton model
_model = None

TEXT_DIM = 384


def _get_model() -> SentenceTransformer:
    """Get or create the SentenceTransformer model singleton."""
    global _model
    if _model is None:
        logger.info("Loading all-MiniLM-L6-v2 locally (first call only)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("all-MiniLM-L6-v2 loaded successfully")
    return _model


def extract_text_embedding(text: str) -> np.ndarray:
    """
    Extract text embedding using local SentenceTransformer.
    Returns: 384-dim float32 numpy array, L2-normalized.
    """
    if not text or not text.strip():
        return np.zeros(TEXT_DIM, dtype=np.float32)

    try:
        model = _get_model()
        embedding = model.encode(text, convert_to_numpy=True).astype(np.float32)

        # Ensure correct dimension
        if embedding.ndim > 1:
            embedding = embedding.mean(axis=0)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    except Exception as e:
        logger.warning(f"Local text embedding failed: {e}")
        return np.zeros(TEXT_DIM, dtype=np.float32)

def extract_text_embeddings_batch(texts: list) -> np.ndarray:
    """
    Extract text embeddings for a batch of strings.
    Returns: (N, 384) float32 numpy array, L2-normalized.
    """
    if not texts:
        return np.zeros((0, TEXT_DIM), dtype=np.float32)

    try:
        model = _get_model()
        embeddings = model.encode(texts, convert_to_numpy=True).astype(np.float32)
        
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid div by zero
        embeddings = embeddings / norms
        
        return embeddings

    except Exception as e:
        logger.warning(f"Batch local text embedding failed: {e}")
        return np.zeros((len(texts), TEXT_DIM), dtype=np.float32)


def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
