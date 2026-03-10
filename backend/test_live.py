"""
Test the live Wellness Engine with your own images.

Usage:
    python test_live.py image1.png image2.jpg image3.png
    python test_live.py                              # uses a synthetic test image
"""

import sys
import requests
import base64
import json
from PIL import Image
import io

API_URL = "https://wellnees-engine-production-20e7.up.railway.app"


def image_to_base64(path: str) -> str:
    """Load an image file and convert to base64."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        # Downscale if too large (saves bandwidth)
        if max(img.size) > 800:
            img.thumbnail((800, 800))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_synthetic_image() -> str:
    """Create a synthetic test image."""
    img = Image.new("RGB", (320, 480), color=(30, 30, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=50)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# Build posts from command line args or synthetic image
if len(sys.argv) > 1:
    image_paths = sys.argv[1:]
    print(f"Using {len(image_paths)} image(s): {image_paths}")
    posts = []
    for i, path in enumerate(image_paths):
        posts.append({
            "post_id": f"post-{i+1}",
            "image_base64": image_to_base64(path),
            "dwell_time_seconds": 5.0,
            "timestamp_utc": "2026-02-28T00:00:00Z"
        })
else:
    print("No images provided, using synthetic test image.")
    print("Usage: python test_live.py image1.png image2.jpg ...")
    posts = [{
        "post_id": "post-1",
        "image_base64": make_synthetic_image(),
        "dwell_time_seconds": 5.0,
        "timestamp_utc": "2026-02-28T00:00:00Z"
    }]

payload = {
    "user_id": "test-user-001",
    "session_id": "test-session-custom",
    "posts": posts,
    "session_start_utc": "2026-02-28T00:00:00Z",
    "session_end_utc": "2026-02-28T00:05:00Z",
    "circadian_score": 0.3,
    "night_session_duration_minutes": 0
}

print(f"\nSending {len(posts)} post(s) to {API_URL}/api/analyze ...")
r = requests.post(f"{API_URL}/api/analyze", json=payload, timeout=300)

print(f"\nStatus: {r.status_code}")
print(f"Response:\n{json.dumps(r.json(), indent=2)}")
