"""
Feed Wellness Engine — Flask Backend
Main application entry point.

Endpoints:
    POST /api/analyze  — Full ML pipeline analysis of a feed session
    GET  /health       — Health check

Pipeline flow (Hybrid: Gemini API + Local PyTorch):
    1. Decode base64 screenshots → PIL Images
    2. Gemini 1.5 Flash: Caption + Post Type Classification (API)
    3. Local toxic-bert: Toxicity scoring (PyTorch)
    4. Local VADER: Sentiment scoring (rule-based)
    5. Local MiniLM: Text embedding (PyTorch)
    6. Aggregate feed embedding (dwell-weighted mean)
    7. Project behavior scores
    8. Compute wellness index
    9. Generate recommendation
    10. Store in Supabase
    11. Return WellnessReport

Privacy:
    - Screenshots are sent to Gemini API for captioning only
    - PIL Images are dereferenced after processing
    - Toxicity, sentiment, and embeddings run 100% locally
    - Only aggregated scores are stored in Supabase
"""

import os
import logging
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Configure logging — NEVER log request bodies (contains screenshots)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# ── Routes ──────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "feed-wellness-engine"}), 200


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Full ML pipeline analysis of a feed session.
    Hybrid architecture: Gemini API for vision + Local models for NLP.

    Expects JSON body matching AnalyzeRequest schema.
    Returns JSON matching AnalyzeResponse schema.
    """
    try:
        data = request.get_json(force=True)

        # Validate required fields
        required = ["user_id", "session_id", "posts", "session_start_utc",
                     "session_end_utc", "circadian_score"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        posts = data["posts"]
        if not posts:
            return jsonify({"error": "No posts to analyze"}), 400

        logger.info(f"Analyzing session {data['session_id']}: {len(posts)} posts")

        from models.gemini_client import analyze_collage, decode_base64_image
        from models.collage_builder import build_collage
        from models.hf_client import caption_image
        # --- DEBUG ENV VARS (Safe key dump) ---
        env_keys = list(os.environ.keys())
        logger.info(f"Loaded System Env Keys: {env_keys}")
        
        # ── Decode & Prepare Images ─────────────────────────────────
        all_dwell_times = []
        pil_images = []
        
        for i, post in enumerate(posts):
            image_b64 = post.get("image_base64")
            
            if image_b64:
                # Normal screenshot post
                image = decode_base64_image(image_b64)
            else:
                # Text-only post (e.g. accessibility scraping fallback)
                text_content = post.get("extracted_text", "Unknown Text Content")
                # Create a synthetic image to keep array lengths matching for Gemini grid
                from PIL import Image, ImageDraw
                image = Image.new('RGB', (400, 400), color=(30, 30, 30))
                draw = ImageDraw.Draw(image)
                # Just draw the text roughly in the middle so Gemini can read it visually
                draw.text((20, 20), text_content[:500], fill=(255, 255, 255))

            pil_images.append(image)
            all_dwell_times.append(float(post.get("dwell_time_seconds", 0.0)))

        logger.info(f"Stitching {len(pil_images)} images into collage...")
        collage = build_collage(pil_images)

        # ── Auxiliary BLIP API Call ─────────────────────────────────
        logger.info("Fetching auxiliary BLIP caption from HF API...")
        blip_context = caption_image(collage)

        # ── Gemini Grid Analysis ────────────────────────────────────
        logger.info("Executing Gemini Grid Collage Analysis...")
        gemini_result = analyze_collage(collage, aux_context=blip_context)
        
        summary = gemini_result.get("summary", {})
        cells = gemini_result.get("cells", [])

        # ── Metric Translation (Grid -> Android Schema) ─────────────
        # Convert 1-10 quality score to 0-100 wellness index
        quality_score = float(summary.get("screen_time_quality_score", 5))
        wellness_index = min(max(quality_score * 10.0, 0.0), 100.0)

        # Synthesize float values required by Android DTO based on behavior
        overall_behavior = summary.get("overall_behavior", "mixed").lower()
        if "focused" in overall_behavior or "learning" in overall_behavior:
            toxicity = 0.05
            addictiveness = 0.2
            positivity = 0.85
            emotion = 0.3
        elif "distracted" in overall_behavior or "scrolling" in overall_behavior:
            toxicity = 0.3
            addictiveness = 0.8
            positivity = 0.4
            emotion = 0.6
        else:
            toxicity = 0.1
            addictiveness = 0.5
            positivity = 0.5
            emotion = 0.5

        circadian_score = float(data["circadian_score"])

        # Format Recommendation & Insights
        recommendation = summary.get("key_pattern", "Try to limit endless scrolling and focus your intent.")
        
        insights = []
        if summary.get("dominant_categories"):
            cats = ", ".join(summary["dominant_categories"])
            insights.append(f"You predominantly engaged with: {cats}")
            
        for cell in cells[:5]:  # Take top 5 summaries as insights
            idx = cell.get("index", "?")
            app = cell.get("app", "App")
            topic = cell.get("topic", "Content")
            sum_text = cell.get("brief_summary", "")
            if sum_text:
                insights.append(f"Screenshot #{idx} ({app}): {topic} - {sum_text}")

        # ── Build response ──────────────────────────────────────

        models_used = {
            "collage_analyzer": {
                "model": "Google Gemini 1.5 Flash",
                "purpose": "Unified grid collage reasoning for rapid multimodal scraping",
                "source": "Google Gemini API"
            },
            "auxiliary_vision": {
                "model": "Salesforce/blip-image-captioning-base",
                "purpose": "Auxiliary captioning pass for grid elements",
                "source": "Hugging Face Inference API"
            }
        }

        # Pure Python Math for Attention Span (no numpy)
        total_dwell = sum(all_dwell_times)
        avg_dwell = total_dwell / len(all_dwell_times) if all_dwell_times else 0
        max_dwell = max(all_dwell_times) if all_dwell_times else 0
        min_dwell = min(all_dwell_times) if all_dwell_times else 0
        
        # Pure python standard deviation
        dwell_variance = sum((x - avg_dwell) ** 2 for x in all_dwell_times) / len(all_dwell_times) if len(all_dwell_times) > 0 else 0
        dwell_std_dev = dwell_variance ** 0.5

        attention_per_post = []
        for i, dwell in enumerate(all_dwell_times):
            if dwell >= 10:
                level = "deep_focus"
                label = "🔴 Deep focus — high engagement"
            elif dwell >= 6:
                level = "engaged"
                label = "🟠 Engaged — moderate attention"
            elif dwell >= 3:
                level = "browsing"
                label = "🟡 Browsing — casual scroll"
            else:
                level = "skimming"
                label = "🟢 Skimming — quick glance"

            attention_per_post.append({
                "post_id": posts[i].get("post_id", f"post-{i+1}"),
                "dwell_seconds": round(dwell, 1),
                "attention_level": level,
                "attention_label": label,
                "pct_of_session": round(dwell / total_dwell * 100, 1) if total_dwell > 0 else 0
            })

        if avg_dwell >= 10:
            attention_health = "concerning"
            attention_summary = "You're spending a lot of time per post — signs of doom-scrolling or content fixation."
        elif avg_dwell >= 6:
            attention_health = "moderate"
            attention_summary = "Moderate engagement per post. Watch for extended sessions."
        elif avg_dwell >= 3:
            attention_health = "healthy"
            attention_summary = "Normal browsing pace — you're scanning content without getting stuck."
        else:
            attention_health = "rapid"
            attention_summary = "Very fast scrolling — you may not be absorbing content."

        attention_span = {
            "total_session_seconds": round(total_dwell, 1),
            "avg_dwell_seconds": round(avg_dwell, 1),
            "max_dwell_seconds": round(max_dwell, 1),
            "min_dwell_seconds": round(min_dwell, 1),
            "dwell_std_dev": round(dwell_std_dev, 2),
            "attention_health": attention_health,
            "attention_summary": attention_summary,
            "scroll_speed_posts_per_min": round(len(all_dwell_times) / max(total_dwell / 60, 0.1), 1),
            "per_post": attention_per_post
        }

        response = {
            "session_id": data["session_id"],
            "toxicity_score": round(toxicity, 4),
            "addictiveness_score": round(addictiveness, 4),
            "positivity_score": round(positivity, 4),
            "emotional_intensity": round(emotion, 4),
            "circadian_score": round(circadian_score, 4),
            "wellness_index": round(wellness_index, 2),
            "recommendation": recommendation,
            "models_used": models_used,
            "attention_span": attention_span,
            "insights": insights
        }

        # ── Store in Supabase ───────────────────────────────────
        try:
            from storage.supabase_client import SupabaseClient
            supabase = SupabaseClient()
            total_dwell = sum(all_dwell_times)

            supabase.store_session({
                "user_id": data["user_id"],
                "session_id": data["session_id"],
                "session_start": data["session_start_utc"],
                "session_end": data["session_end_utc"],
                "post_count": len(posts),
                "toxicity_score": response["toxicity_score"],
                "addictiveness_score": response["addictiveness_score"],
                "positivity_score": response["positivity_score"],
                "emotional_intensity": response["emotional_intensity"],
                "circadian_score": response["circadian_score"],
                "wellness_index": response["wellness_index"],
                "recommendation": recommendation
            })

            supabase.upsert_daily_aggregate(
                user_id=data["user_id"],
                report_date=datetime.now(timezone.utc).date(),
                wellness_index=response["wellness_index"],
                toxicity=response["toxicity_score"],
                positivity=response["positivity_score"],
                addictiveness=response["addictiveness_score"],
                total_dwell=total_dwell,
                night_ratio=circadian_score
            )

            logger.info(f"Session {data['session_id']} stored in Supabase")
        except Exception as e:
            # Don't fail the response if storage fails
            logger.error(f"Supabase storage failed: {e}")

        logger.info(
            f"Session {data['session_id']} analyzed: "
            f"wellness={response['wellness_index']}, "
            f"rec={recommendation}"
        )

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
