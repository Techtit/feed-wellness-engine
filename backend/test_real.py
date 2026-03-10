"""
Test the Wellness Engine with real images — shows ALL data returned.

Usage: python test_real.py test_images\Capture.PNG test_images\Captureik.PNG test_images\Capturejjn.PNG
"""

import sys
import requests
import base64
import json
import io
from PIL import Image

API_URL = "https://wellnees-engine-production-20e7.up.railway.app"


def load_and_encode(path: str) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if max(img.size) > 800:
            img.thumbnail((800, 800))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# Build posts
image_files = sys.argv[1:] if len(sys.argv) > 1 else [
    "test_images\\Capture.PNG", "test_images\\Captureik.PNG", "test_images\\Capturejjn.PNG"
]

# Varied dwell times to simulate real attention patterns
dwell_times = [8.5, 4.2, 12.3]  # seconds per post

posts = []
for i, path in enumerate(image_files):
    print(f"  Loading: {path}")
    posts.append({
        "post_id": f"post-{i+1}",
        "image_base64": load_and_encode(path),
        "dwell_time_seconds": dwell_times[i % len(dwell_times)],
        "timestamp_utc": f"2026-03-09T15:2{i}:00Z"
    })

payload = {
    "user_id": "aaryan-test",
    "session_id": "real-session-v2",
    "posts": posts,
    "session_start_utc": "2026-03-09T15:20:00Z",
    "session_end_utc": "2026-03-09T15:30:00Z",
    "circadian_score": 0.1,  # Afternoon session (low circadian risk)
    "night_session_duration_minutes": 0
}

print(f"\n📡 Sending {len(posts)} posts to {API_URL}/api/analyze ...")
r = requests.post(f"{API_URL}/api/analyze", json=payload, timeout=300)
result = r.json()

if r.status_code != 200:
    print(f"❌ Error {r.status_code}: {json.dumps(result, indent=2)}")
    sys.exit(1)

# ── Print ALL returned data ──────────────────────────────────────
print(f"\n{'='*60}")
print(f"  WELLNESS ENGINE — FULL ANALYSIS REPORT")
print(f"{'='*60}")

# Core scores
print(f"\n📊 CORE SCORES")
print(f"  Wellness Index:      {result['wellness_index']}/100")
print(f"  Toxicity:            {result['toxicity_score']}")
print(f"  Addictiveness:       {result['addictiveness_score']}")
print(f"  Positivity:          {result['positivity_score']}")
print(f"  Emotional Intensity: {result['emotional_intensity']}")
print(f"  Circadian Score:     {result['circadian_score']}")
print(f"  Recommendation:      {result['recommendation']}")

# Models used
if "models_used" in result:
    print(f"\n🤖 MODELS USED")
    for key, info in result["models_used"].items():
        print(f"  {key}:")
        print(f"    Model:   {info['model']}")
        print(f"    Purpose: {info['purpose']}")
        print(f"    Source:  {info['source']}")

# Attention span
if "attention_span" in result:
    att = result["attention_span"]
    print(f"\n⏱️  ATTENTION SPAN ANALYSIS")
    print(f"  Total session:       {att['total_session_seconds']}s")
    print(f"  Avg dwell/post:      {att['avg_dwell_seconds']}s")
    print(f"  Max dwell:           {att['max_dwell_seconds']}s")
    print(f"  Min dwell:           {att['min_dwell_seconds']}s")
    print(f"  Std deviation:       {att['dwell_std_dev']}s")
    print(f"  Scroll speed:        {att['scroll_speed_posts_per_min']} posts/min")
    print(f"  Attention health:    {att['attention_health']}")
    print(f"  Summary:             {att['attention_summary']}")
    print(f"\n  Per-post attention:")
    for p in att.get("per_post", []):
        print(f"    {p['post_id']}: {p['dwell_seconds']}s — {p['attention_label']} ({p['pct_of_session']}% of session)")

# Insights
if "insights" in result:
    ins = result["insights"]

    # Feed health grade
    grade = ins.get("feed_health_grade", {})
    print(f"\n📝 FEED HEALTH GRADE")
    print(f"  Grade:       {grade.get('grade', '?')} ({grade.get('label', '?')})")
    print(f"  Score:       {grade.get('score', '?')}/100")
    print(f"  Explanation: {grade.get('explanation', '')}")
    if grade.get("penalties"):
        print(f"  Penalties:   {', '.join(grade['penalties'])}")

    # Mood analysis
    mood = ins.get("mood_analysis", {})
    print(f"\n😊 MOOD ANALYSIS")
    print(f"  Mood:            {mood.get('mood_emoji', '')} {mood.get('overall_mood', '?')}")
    print(f"  Description:     {mood.get('description', '')}")
    print(f"  Positivity:      {mood.get('positivity_level', '?')}")
    print(f"  Emotional drain: {mood.get('emotional_drain', '?')} (score: {mood.get('emotional_drain_score', '?')})")

    # Content themes
    themes = ins.get("content_themes", {})
    print(f"\n🎯 CONTENT THEMES")
    print(f"  Dominant post type: {themes.get('dominant_post_type', '?')}")
    print(f"  Dominant theme:     {themes.get('dominant_theme', '?')}")
    if themes.get("detected_themes"):
        for t in themes["detected_themes"]:
            print(f"    - {t['theme']} (confidence: {t['confidence']}, matches: {t['keyword_matches']})")
    if themes.get("post_type_distribution"):
        print(f"  Type distribution:  {themes['post_type_distribution']}")

    # Feed patterns
    patterns = ins.get("feed_patterns", {})
    print(f"\n📈 FEED PATTERNS")
    print(f"  Emotional trajectory: {patterns.get('emotional_trajectory', '?')}")
    print(f"  Content variety:      {patterns.get('content_variety_score', '?')}")
    print(f"  Content loop:         {'⚠️ YES' if patterns.get('content_loop_detected') else '✅ No'}")

    # Time analysis
    time_a = ins.get("time_analysis", {})
    print(f"\n🕐 TIME ANALYSIS")
    print(f"  Circadian risk:  {time_a.get('circadian_risk_level', '?')}")
    print(f"  Night browsing:  {'Yes' if time_a.get('night_browsing') else 'No'}")
    if time_a.get("warning"):
        print(f"  ⚠️ Warning:     {time_a['warning']}")

    # Risk flags
    risks = ins.get("risk_flags", [])
    if risks:
        print(f"\n🚩 RISK FLAGS")
        for r in risks:
            sev_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(r["severity"], "⚪")
            print(f"  {sev_icon} [{r['severity'].upper()}] {r['type']}: {r['message']}")

    # Actionable insights
    actions = ins.get("actionable_insights", [])
    if actions:
        print(f"\n💡 ACTIONABLE INSIGHTS")
        for a in actions:
            pri = {"high": "🔴", "medium": "🟡", "low": "🟢", "positive": "✅"}.get(a["priority"], "⚪")
            print(f"  {pri} [{a['priority'].upper()}] {a['action']}")
            print(f"     Reason: {a['reason']}")
            print(f"     Impact: {a['impact']}")

    # Per-post breakdown
    per_post = ins.get("per_post", [])
    if per_post:
        print(f"\n📋 PER-POST BREAKDOWN")
        for p in per_post:
            print(f"  {p['post_id']}:")
            print(f"    Caption:      {p.get('caption', '(none)')[:80]}")
            print(f"    Type:         {p.get('dominant_type', '?')}")
            print(f"    Toxicity:     {p.get('toxicity', 0)}")
            print(f"    Positivity:   {p.get('positivity', 0)}")
            print(f"    Emotion:      {p.get('emotional_intensity', 0)}")
            print(f"    Dwell time:   {p.get('dwell_time', 0)}s")

print(f"\n{'='*60}")
print(f"  Full JSON saved to: test_result.json")
print(f"{'='*60}")

with open("test_result.json", "w") as f:
    json.dump(result, f, indent=2)
