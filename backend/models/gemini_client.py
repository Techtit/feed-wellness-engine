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
    """Decode base64 string to PIL Image, handling data URI prefixes."""
    # Strip data URI prefix if present (e.g., 'data:image/jpeg;base64,')
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def analyze_collage(collage: Image.Image, aux_context: str = "") -> Dict[str, Any]:
    """
    Analyze a grid collage image of multiple screenshots using Gemini 1.5 Flash.
    """
    model = _get_model()

    prompt = f"""You have to analyse a collage image containing multiple mobile screenshots arranged in a grid (2x2, 3x3, or 4x4).

Each grid cell represents a separate user screen, mostly from social media apps.

Your job is to extract structured behavioral insights.

INSTRUCTIONS:

1. Treat each grid cell as an independent screenshot. The cells are numbered in the top-left with a tiny red square.
2. Do NOT mix contexts between cells.
3. If text is unclear, infer from visual patterns (UI, thumbnails, layout).
4. Be concise and structured.

FOR EACH GRID CELL RETURN:

* index: (1 to N, left to right, top to bottom)
* app: (Instagram, Twitter/X, YouTube, Unknown, etc.)
* content_type: (reel, post, meme, story, feed, article, etc.)
* topic: (crypto, entertainment, fitness, news, etc.)
* engagement_level: (low, medium, high)
* intent: (learning, scrolling, entertainment, distraction, productivity)
* brief_summary: (max 10 words)

FINAL OUTPUT:

* dominant_categories: [top 3 topics]
* overall_behavior: (focused, distracted, mixed)
* screen_time_quality_score: (1-10)
* key_pattern: (1 sentence insight about usage pattern)

OUTPUT FORMAT (STRICT JSON ONLY):

{{
"cells": [
{{
"index": 1,
"app": "",
"content_type": "",
"topic": "",
"engagement_level": "",
"intent": "",
"brief_summary": ""
}}
],
"summary": {{
"dominant_categories": [],
"overall_behavior": "",
"screen_time_quality_score": 0,
"key_pattern": ""
}}
}}

IMPORTANT:

* Do not add extra text outside JSON.
* Do not hallucinate specific names if unclear.
* Keep outputs consistent and minimal.

Optional additional context from another model: {aux_context}
"""

    try:
        response = model.generate_content([prompt, collage])
        response_text = response.text.strip()

        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1]).strip()

        result = json.loads(response_text)
        logger.info(f"Collage processed successfully. Extracted {len(result.get('cells', []))} cells.")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Gemini JSON parse failed: {e}. Raw: {response_text[:200]}")
        return _fallback_result()
    except Exception as e:
        logger.error(f"Gemini analysis failed: {e}")
        return _fallback_result()


def _fallback_result() -> Dict[str, Any]:
    """Return safe fallback if Gemini fails."""
    return {
        "cells": [],
        "summary": {
            "dominant_categories": [],
            "overall_behavior": "unknown",
            "screen_time_quality_score": 5,
            "key_pattern": "Pipeline error occurred."
        }
    }
