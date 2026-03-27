"""
Gemini Vision Client
Uses Google Gemini 1.5 Flash for multimodal image analysis.

Handles:
  - Image captioning (text extraction from screenshots)
  - Post type classification (zero-shot into video/reel/static/text/meme/news)

Requires GEMINI_API_KEY env var.
"""

import os
import io
import base64
import json
import logging
from PIL import Image
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Singleton client
_model = None

# Post type categories
POST_CATEGORIES = ["video", "reel", "static", "text", "meme", "news"]


def _get_model():
    """Get or create the Gemini GenerativeModel singleton."""
    global _model
    if _model is None:
        import google.generativeai as genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Get a free key at "
                "https://aistudio.google.com/app/apikey"
            )
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel("gemini-1.5-flash")
        logger.info("Gemini 1.5 Flash model initialized")
    return _model


def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def analyze_image(image: Image.Image) -> Dict[str, Any]:
    """
    Analyze a post screenshot using Gemini 1.5 Flash.

    Returns a dict with:
      - caption: str — detailed text description of the image
      - post_type: dict — probability distribution over post categories
      - dominant_type: str — the most likely post type
    """
    model = _get_model()

    prompt = f"""Analyze this social media post screenshot. Return a JSON object with exactly these fields:

1. "caption": A detailed text description of the image content, including any visible text, people, objects, emotions, and themes. Be descriptive (2-3 sentences).

2. "post_type": A JSON object with probability scores (0.0 to 1.0, summing to 1.0) for each category: {json.dumps(POST_CATEGORIES)}. Estimate how likely this post is each type.

Return ONLY valid JSON, no markdown formatting, no code blocks. Example format:
{{"caption": "A person standing in front of a sunset with text overlay saying motivational quote", "post_type": {{"video": 0.05, "reel": 0.1, "static": 0.6, "text": 0.1, "meme": 0.1, "news": 0.05}}}}"""

    try:
        response = model.generate_content([prompt, image])
        response_text = response.text.strip()

        # Clean up response — remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Remove first and last lines (```json and ```)
            response_text = "\n".join(lines[1:-1]).strip()

        result = json.loads(response_text)

        caption = result.get("caption", "")
        post_type_raw = result.get("post_type", {})

        # Normalize post type distribution
        post_type = {}
        total = sum(post_type_raw.values()) if post_type_raw else 1.0
        for cat in POST_CATEGORIES:
            score = float(post_type_raw.get(cat, 0.0))
            post_type[cat] = round(score / max(total, 1e-6), 4)

        dominant_type = max(post_type, key=post_type.get) if post_type else "static"

        logger.info(f"Gemini caption: {caption[:80]}...")
        logger.info(f"Gemini post type: {dominant_type} ({post_type})")

        return {
            "caption": caption,
            "post_type": post_type,
            "dominant_type": dominant_type
        }

    except json.JSONDecodeError as e:
        logger.error(f"Gemini JSON parse failed: {e}. Raw: {response_text[:200]}")
        return _fallback_result()
    except Exception as e:
        logger.error(f"Gemini analysis failed: {e}")
        return _fallback_result()


def _fallback_result() -> Dict[str, Any]:
    """Return safe fallback if Gemini fails."""
    uniform = {cat: round(1.0 / len(POST_CATEGORIES), 4) for cat in POST_CATEGORIES}
    return {
        "caption": "",
        "post_type": uniform,
        "dominant_type": "static"
    }
