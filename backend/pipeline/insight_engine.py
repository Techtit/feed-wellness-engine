"""
Insight Engine
Generates human-readable, actionable insights from per-post analysis data.

Provides:
- Per-post breakdown (caption, classification, scores)
- Content theme detection
- Feed pattern analysis
- Mood trajectory
- Late-night / circadian warnings
- Engagement hook detection
- Actionable recommendations with reasoning
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime


def generate_insights(
    per_post_data: List[Dict[str, Any]],
    behavior_scores: Dict[str, float],
    wellness_index: float,
    circadian_score: float,
    recommendation: str,
    session_start_utc: str,
    session_end_utc: str
) -> Dict[str, Any]:
    """
    Generate rich insights from the analyzed feed session.

    Returns a dict with:
        - per_post: detailed breakdown per post
        - content_themes: detected content themes/categories
        - feed_patterns: patterns in the feed (emotional trajectory, engagement hooks)
        - mood_analysis: overall mood assessment
        - time_analysis: circadian/time-of-day insights
        - actionable_insights: human-readable recommendations with reasoning
        - feed_health_grade: A-F grade with explanation
        - risk_flags: any concerning patterns detected
    """

    insights = {}

    # ── Per-post breakdown ───────────────────────────────────────
    insights["per_post"] = per_post_data

    # ── Content theme detection ──────────────────────────────────
    insights["content_themes"] = _detect_themes(per_post_data)

    # ── Feed patterns ────────────────────────────────────────────
    insights["feed_patterns"] = _analyze_feed_patterns(per_post_data)

    # ── Mood analysis ────────────────────────────────────────────
    insights["mood_analysis"] = _analyze_mood(per_post_data, behavior_scores)

    # ── Time analysis ────────────────────────────────────────────
    insights["time_analysis"] = _analyze_time(
        circadian_score, session_start_utc, session_end_utc, per_post_data
    )

    # ── Feed health grade ────────────────────────────────────────
    insights["feed_health_grade"] = _compute_grade(wellness_index, behavior_scores, circadian_score)

    # ── Risk flags ───────────────────────────────────────────────
    insights["risk_flags"] = _detect_risks(behavior_scores, circadian_score, per_post_data)

    # ── Actionable insights ──────────────────────────────────────
    insights["actionable_insights"] = _generate_actionable_insights(
        behavior_scores, wellness_index, circadian_score, per_post_data, insights
    )

    return insights


def _detect_themes(per_post_data: List[Dict]) -> Dict[str, Any]:
    """Detect dominant content themes from post classifications and captions."""
    if not per_post_data:
        return {"dominant_theme": "unknown", "themes": []}

    # Aggregate post type classifications
    type_totals = {}
    captions = []
    for post in per_post_data:
        for label, prob in post.get("post_type", {}).items():
            type_totals[label] = type_totals.get(label, 0) + prob
        captions.append(post.get("caption", "").lower())

    # Average across posts
    n = len(per_post_data)
    type_avgs = {k: round(v / n, 3) for k, v in type_totals.items()}
    dominant_type = max(type_avgs, key=type_avgs.get) if type_avgs else "unknown"

    # Theme detection from captions
    theme_keywords = {
        "relationship": ["relationship", "love", "marry", "girl", "boy", "her", "him",
                         "dreaming", "miss", "over", "breakup", "broke"],
        "humor": ["meme", "funny", "lol", "lmao", "joke", "bro"],
        "motivation": ["hustle", "grind", "success", "sigma", "stoic", "discipline"],
        "emotional": ["feel", "cry", "sad", "pain", "hurt", "accepting", "joy",
                       "miss", "dreaming", "heart"],
        "news": ["news", "breaking", "update", "politics", "election"],
        "entertainment": ["movie", "show", "watch", "scene", "character"],
    }

    detected_themes = []
    full_text = " ".join(captions)
    for theme, keywords in theme_keywords.items():
        matches = sum(1 for kw in keywords if kw in full_text)
        if matches >= 1:
            detected_themes.append({
                "theme": theme,
                "confidence": min(matches / 3, 1.0),
                "keyword_matches": matches
            })

    detected_themes.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "dominant_post_type": dominant_type,
        "post_type_distribution": type_avgs,
        "detected_themes": detected_themes,
        "dominant_theme": detected_themes[0]["theme"] if detected_themes else "general"
    }


def _analyze_feed_patterns(per_post_data: List[Dict]) -> Dict[str, Any]:
    """Analyze patterns across the feed: emotional trajectory, engagement hooks."""
    if not per_post_data:
        return {}

    # Emotional trajectory (are emotions escalating?)
    emotions = [p.get("emotional_intensity", 0.5) for p in per_post_data]
    toxicities = [p.get("toxicity", 0.0) for p in per_post_data]

    # Trend detection
    if len(emotions) >= 2:
        emotion_trend = emotions[-1] - emotions[0]
        if emotion_trend > 0.1:
            trajectory = "escalating"
        elif emotion_trend < -0.1:
            trajectory = "de-escalating"
        else:
            trajectory = "stable"
    else:
        trajectory = "insufficient_data"

    # Content variety (are you stuck in a loop?)
    captions = [p.get("caption", "") for p in per_post_data]
    unique_themes = len(set(c[:30] for c in captions))  # rough uniqueness
    variety_score = unique_themes / max(len(captions), 1)

    # Engagement hooks
    dwell_times = [p.get("dwell_time", 0) for p in per_post_data]
    avg_dwell = np.mean(dwell_times) if dwell_times else 0
    max_dwell = max(dwell_times) if dwell_times else 0

    return {
        "emotional_trajectory": trajectory,
        "emotion_trend_delta": round(emotions[-1] - emotions[0], 3) if len(emotions) >= 2 else 0,
        "content_variety_score": round(variety_score, 2),
        "avg_dwell_seconds": round(avg_dwell, 1),
        "max_dwell_seconds": round(max_dwell, 1),
        "total_posts_analyzed": len(per_post_data),
        "content_loop_detected": variety_score < 0.4 and len(per_post_data) >= 3
    }


def _analyze_mood(per_post_data: List[Dict], behavior_scores: Dict) -> Dict[str, Any]:
    """Assess overall mood of the feed session."""
    positivity = behavior_scores.get("positivity", 0.5)
    toxicity = behavior_scores.get("toxicity", 0.0)
    emotional_intensity = behavior_scores.get("emotional_intensity", 0.5)

    # Mood classification
    if positivity >= 0.7:
        mood = "positive"
        emoji = "😊"
        description = "Your feed is uplifting, with positive and encouraging content."
    elif positivity >= 0.5 and toxicity < 0.2:
        mood = "neutral"
        emoji = "😐"
        description = "Your feed is emotionally neutral — not particularly uplifting or draining."
    elif toxicity >= 0.5:
        mood = "toxic"
        emoji = "⚠️"
        description = "Your feed contains high toxicity. This can negatively impact mental health."
    elif positivity < 0.3:
        mood = "negative"
        emoji = "😔"
        description = "Your feed has a negative emotional tone. Consider diversifying content."
    elif emotional_intensity >= 0.7:
        mood = "emotionally_charged"
        emoji = "💔"
        description = "Your feed is emotionally intense, which can be draining over time."
    else:
        mood = "mixed"
        emoji = "🤷"
        description = "Your feed has mixed emotional signals."

    # Emotional drain estimate
    drain_score = (1 - positivity) * 0.4 + toxicity * 0.3 + emotional_intensity * 0.3
    if drain_score > 0.6:
        drain_level = "high"
    elif drain_score > 0.35:
        drain_level = "moderate"
    else:
        drain_level = "low"

    return {
        "overall_mood": mood,
        "mood_emoji": emoji,
        "description": description,
        "positivity_level": _level_label(positivity),
        "emotional_drain": drain_level,
        "emotional_drain_score": round(drain_score, 3)
    }


def _analyze_time(circadian_score: float, start: str, end: str,
                   per_post_data: List[Dict]) -> Dict[str, Any]:
    """Analyze time-of-day and circadian impact."""
    # Determine time of day category
    if circadian_score >= 0.8:
        time_risk = "critical"
        time_warning = "You're scrolling deep into the night. Blue light and emotional content at this hour significantly disrupts sleep quality."
    elif circadian_score >= 0.5:
        time_risk = "high"
        time_warning = "Late-night scrolling detected. This can delay melatonin production and affect tomorrow's energy."
    elif circadian_score >= 0.2:
        time_risk = "moderate"
        time_warning = "Evening browsing. Consider winding down soon for better sleep."
    else:
        time_risk = "low"
        time_warning = None

    total_dwell = sum(p.get("dwell_time", 0) for p in per_post_data)

    return {
        "circadian_risk_level": time_risk,
        "warning": time_warning,
        "session_duration_seconds": round(total_dwell, 1),
        "posts_per_minute": round(len(per_post_data) / max(total_dwell / 60, 0.1), 1),
        "night_browsing": circadian_score >= 0.5
    }


def _compute_grade(wellness_index: float, behavior_scores: Dict,
                   circadian_score: float) -> Dict[str, str]:
    """Compute a letter grade for the feed session."""
    if wellness_index >= 80:
        grade, label = "A", "Excellent"
    elif wellness_index >= 65:
        grade, label = "B", "Good"
    elif wellness_index >= 50:
        grade, label = "C", "Fair"
    elif wellness_index >= 35:
        grade, label = "D", "Poor"
    else:
        grade, label = "F", "Unhealthy"

    # Penalty explanations
    penalties = []
    if circadian_score >= 0.5:
        penalties.append(f"Late-night use (-{round(circadian_score * 15, 1)}pts est.)")
    if behavior_scores.get("toxicity", 0) >= 0.3:
        penalties.append(f"Toxic content (-{round(behavior_scores['toxicity'] * 25, 1)}pts est.)")
    if behavior_scores.get("addictiveness", 0) >= 0.5:
        penalties.append(f"Addictive patterns (-{round(behavior_scores['addictiveness'] * 20, 1)}pts est.)")
    if behavior_scores.get("positivity", 0.5) < 0.3:
        penalties.append("Low positivity")

    return {
        "grade": grade,
        "label": label,
        "score": round(wellness_index, 1),
        "penalties": penalties,
        "explanation": f"Your feed earned a {grade} ({label}). "
                       + (f"Main drags: {', '.join(penalties)}." if penalties else "Keep it up!")
    }


def _detect_risks(behavior_scores: Dict, circadian_score: float,
                   per_post_data: List[Dict]) -> List[Dict[str, str]]:
    """Detect concerning patterns that warrant attention."""
    risks = []

    if circadian_score >= 0.7:
        risks.append({
            "type": "sleep_disruption",
            "severity": "high",
            "message": "Scrolling past midnight disrupts circadian rhythm and melatonin production."
        })

    if behavior_scores.get("toxicity", 0) >= 0.5:
        risks.append({
            "type": "toxic_exposure",
            "severity": "high",
            "message": "Sustained toxic content exposure is linked to increased anxiety and negativity."
        })

    if behavior_scores.get("addictiveness", 0) >= 0.6:
        risks.append({
            "type": "addictive_loop",
            "severity": "medium",
            "message": "Your feed is designed to keep you scrolling. Short-form video content triggers dopamine loops."
        })

    if behavior_scores.get("positivity", 0.5) < 0.3 and behavior_scores.get("emotional_intensity", 0.5) > 0.6:
        risks.append({
            "type": "emotional_drain",
            "severity": "medium",
            "message": "Low positivity + high emotional intensity can lead to mood degradation."
        })

    # Content loop detection
    captions = [p.get("caption", "") for p in per_post_data]
    if len(captions) >= 3 and len(set(c[:20] for c in captions)) <= len(captions) * 0.5:
        risks.append({
            "type": "content_bubble",
            "severity": "low",
            "message": "Your feed shows repetitive content themes. Algorithm may be creating an echo chamber."
        })

    return risks


def _generate_actionable_insights(
    behavior_scores: Dict, wellness_index: float,
    circadian_score: float, per_post_data: List[Dict],
    insights: Dict
) -> List[Dict[str, str]]:
    """Generate specific, actionable recommendations with reasoning."""
    actions = []

    # Time-based insights
    if circadian_score >= 0.7:
        actions.append({
            "priority": "high",
            "action": "Enable sleep mode or put your phone down",
            "reason": f"It's late at night (circadian score: {circadian_score}). "
                      "Scrolling now delays sleep onset by 30-60 minutes on average.",
            "impact": "Sleep quality +40%"
        })

    # Content-based insights
    themes = insights.get("content_themes", {})
    dominant_theme = themes.get("dominant_theme", "")

    if dominant_theme in ["relationship", "emotional"]:
        actions.append({
            "priority": "medium",
            "action": "Diversify your feed with positive or educational content",
            "reason": f"Your feed is dominated by {dominant_theme} content. "
                      "Sustained emotional content without balance can lower mood.",
            "impact": "Mood improvement +20%"
        })

    if behavior_scores.get("addictiveness", 0) > 0.3:
        actions.append({
            "priority": "medium",
            "action": "Set a 15-minute timer for social media",
            "reason": "The content format (short-form video, reels) is optimized "
                      "for engagement loops. Time-boxing prevents mindless scrolling.",
            "impact": "Screen time -30%"
        })

    if behavior_scores.get("positivity", 0.5) < 0.4:
        actions.append({
            "priority": "low",
            "action": "Follow accounts that post uplifting or educational content",
            "reason": "Your current feed has low positivity. Curating your follows "
                      "directly improves the content algorithm serves you.",
            "impact": "Feed positivity +50%"
        })

    if wellness_index >= 70 and circadian_score < 0.3:
        actions.append({
            "priority": "positive",
            "action": "Your feed looks healthy! Keep it up.",
            "reason": "Good content balance, reasonable browsing time, low toxicity.",
            "impact": "Sustained wellbeing"
        })

    return actions


def _level_label(score: float) -> str:
    if score >= 0.7:
        return "high"
    elif score >= 0.4:
        return "moderate"
    else:
        return "low"
