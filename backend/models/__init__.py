"""Models package — Hybrid architecture: Gemini API + Local PyTorch models."""
from models.gemini_client import analyze_image, decode_base64_image
from models.embedding_engine import extract_text_embedding
