"""
Feed Wellness Engine — Flask Backend
Main application entry point.

Endpoints:
    POST /api/analyze  — Full ML pipeline analysis of a feed session
    GET  /health       — Health check

Pipeline flow (HuggingFace Inference API — zero local model loading):
    1. Decode base64 screenshots → PIL Images
    2. Image captioning (BLIP) → extracted text per post
    3. Post type classification (CLIP zero-shot) per post
    4. Toxicity scoring (toxic-bert) per post
    5. Sentiment scoring (roberta) per post
    6. Text embedding (sentence-transformers) per post
    7. Aggregate feed embedding (dwell-weighted mean)
    8. Project behavior scores
    9. Compute wellness index
    10. Generate recommendation
    11. Store in Supabase
    12. Return WellnessReport

Privacy:
    - Screenshots are sent to HuggingFace API for processing only
    - PIL Images are dereferenced after processing
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
    All ML inference via HuggingFace Inference API — zero local models.

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

        import numpy as np
        from models import hf_client

        # Post type labels
        POST_CATEGORIES = ["video", "reel", "static image", "text post", "meme", "news article"]
        CATEGORY_LABELS = ["video", "reel", "static", "text", "meme", "news"]
        label_map = dict(zip(POST_CATEGORIES, CATEGORY_LABELS))

        # ── Per-post processing ─────────────────────────────────
        all_text_embeddings = []
        all_dwell_times = []
        all_toxicity_scores = []
        all_positivity_scores = []
        all_emotion_scores = []
        all_post_types = []
        per_post_data = []  # Rich per-post breakdown for insights

        for i, post in enumerate(posts):
            logger.info(f"Processing post {i+1}/{len(posts)}...")

            # Decode screenshot
            image = hf_client.decode_base64_image(post["image_base64"])

            # Image captioning → extracted text
            extracted_text = hf_client.caption_image(image)

            # Post type classification (CLIP zero-shot)
            raw_dist = hf_client.classify_image(image, POST_CATEGORIES)
            post_type_dist = {label_map.get(k, k): v for k, v in raw_dist.items()}

            # PRIVACY: Done with image
            del image

            # Toxicity scoring
            toxicity = hf_client.score_toxicity(extracted_text)

            # Sentiment scoring
            positivity, emotional_intensity = hf_client.score_sentiment(extracted_text)

            # Text embedding
            text_emb = hf_client.extract_text_embedding(extracted_text)

            # Collect results
            all_text_embeddings.append(text_emb)
            all_dwell_times.append(post["dwell_time_seconds"])
            all_toxicity_scores.append(toxicity)
            all_positivity_scores.append(positivity)
            all_emotion_scores.append(emotional_intensity)
            all_post_types.append(post_type_dist)

            # Per-post detail for insights
            per_post_data.append({
                "post_id": post.get("post_id", f"post-{i+1}"),
                "caption": extracted_text,
                "post_type": post_type_dist,
                "dominant_type": max(post_type_dist, key=post_type_dist.get) if post_type_dist else "unknown",
                "toxicity": round(toxicity, 4),
                "positivity": round(positivity, 4),
                "emotional_intensity": round(emotional_intensity, 4),
                "dwell_time": post["dwell_time_seconds"]
            })

        # ── Feed-level processing ───────────────────────────────
        from pipeline.feed_aggregator import aggregate_feed
        from pipeline.behavior_projection import project_behavior_scores
        from pipeline.wellness_calculator import compute_wellness_index
        from pipeline.decision_engine import decide_recommendation

        # Aggregate feed embedding (using text embeddings)
        feed_embedding = aggregate_feed(all_text_embeddings, all_dwell_times)

        # Project behavior scores
        behavior_scores = project_behavior_scores(
            feed_embedding=feed_embedding,
            toxicity_scores=all_toxicity_scores,
            positivity_scores=all_positivity_scores,
            emotional_intensity_scores=all_emotion_scores,
            dwell_times=all_dwell_times,
            post_type_distributions=all_post_types
        )

        # Compute wellness index
        circadian_score = data["circadian_score"]
        wellness_index = compute_wellness_index(
            toxicity=behavior_scores["toxicity"],
            addictiveness=behavior_scores["addictiveness"],
            positivity=behavior_scores["positivity"],
            emotional_intensity=behavior_scores["emotional_intensity"],
            circadian_score=circadian_score
        )

        # Generate recommendation
        recommendation = decide_recommendation(
            toxicity=behavior_scores["toxicity"],
            addictiveness=behavior_scores["addictiveness"],
            circadian_score=circadian_score,
            wellness_index=wellness_index
        )

        # ── Generate insights ────────────────────────────────────
        from pipeline.insight_engine import generate_insights

        insights = generate_insights(
            per_post_data=per_post_data,
            behavior_scores=behavior_scores,
            wellness_index=wellness_index,
            circadian_score=circadian_score,
            recommendation=recommendation,
            session_start_utc=data["session_start_utc"],
            session_end_utc=data["session_end_utc"]
        )

        # ── Build response ──────────────────────────────────────

        # Models used in this analysis
        models_used = {
            "image_captioning": {
                "model": "Salesforce/blip-image-captioning-base",
                "purpose": "Extracts text descriptions from post screenshots",
                "source": "HuggingFace Inference API"
            },
            "post_classification": {
                "model": "openai/clip-vit-base-patch32",
                "purpose": "Zero-shot classification into video/reel/static/text/meme/news",
                "source": "HuggingFace Inference API"
            },
            "toxicity_detection": {
                "model": "unitary/toxic-bert",
                "purpose": "Scores text toxicity on 0-1 scale",
                "source": "HuggingFace Inference API"
            },
            "sentiment_analysis": {
                "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "purpose": "Detects positivity and emotional intensity",
                "source": "HuggingFace Inference API"
            },
            "text_embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "purpose": "384-dim text embedding for feed aggregation",
                "source": "HuggingFace Inference API"
            }
        }

        # Attention span analysis
        dwell_arr = np.array(all_dwell_times)
        total_dwell = float(np.sum(dwell_arr))
        avg_dwell = float(np.mean(dwell_arr))
        max_dwell = float(np.max(dwell_arr))
        min_dwell = float(np.min(dwell_arr))

        # Classify attention level per post
        attention_per_post = []
        for i, (dwell, ppd) in enumerate(zip(all_dwell_times, per_post_data)):
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
                "post_id": ppd.get("post_id", f"post-{i+1}"),
                "dwell_seconds": round(dwell, 1),
                "attention_level": level,
                "attention_label": label,
                "pct_of_session": round(dwell / total_dwell * 100, 1) if total_dwell > 0 else 0
            })

        # Overall attention health
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
            "dwell_std_dev": round(float(np.std(dwell_arr)), 2),
            "attention_health": attention_health,
            "attention_summary": attention_summary,
            "scroll_speed_posts_per_min": round(len(all_dwell_times) / max(total_dwell / 60, 0.1), 1),
            "per_post": attention_per_post
        }

        response = {
            "session_id": data["session_id"],
            "toxicity_score": round(behavior_scores["toxicity"], 4),
            "addictiveness_score": round(behavior_scores["addictiveness"], 4),
            "positivity_score": round(behavior_scores["positivity"], 4),
            "emotional_intensity": round(behavior_scores["emotional_intensity"], 4),
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
