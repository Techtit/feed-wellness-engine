import json
import base64
from unittest.mock import patch, MagicMock

# Mock the gemini API to return fixed captions
def mock_analyze_image(image):
    return {
        "caption": "This is a wonderful and beautiful day! I love everything about this. #blessed #happy",
        "post_type": {
            "text": 0.8,
            "image": 0.2,
            "video": 0.0,
            "interactive": 0.0
        },
        "dominant_type": "text"
    }

# Mock Supabase
class MockSupabase:
    def store_session(self, *args, **kwargs):
        print("[MOCK] Stored session to Supabase")
    def upsert_daily_aggregate(self, *args, **kwargs):
        print("[MOCK] Stored daily aggregate to Supabase")

# Apply patches
patch_gemini = patch('models.gemini_client.analyze_image', side_effect=mock_analyze_image)
patch_supabase = patch('storage.supabase_client.SupabaseClient', return_value=MockSupabase())

patch_gemini.start()
patch_supabase.start()

# Now import app
from app import app
client = app.test_client()

b64_img = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="

payload = {
    "user_id": "test_user_123",
    "session_id": "test_session_abc",
    "session_start_utc": "2025-01-01T12:00:00Z",
    "session_end_utc": "2025-01-01T12:10:00Z",
    "circadian_score": 0.8,
    "posts": [
        {
            "post_id": "post_1",
            "image_base64": b64_img,
            "dwell_time_seconds": 8.0
        },
        {
            "post_id": "post_2",
            "image_base64": b64_img,
            "dwell_time_seconds": 12.0
        }
    ]
}

print("Running mock pipeline test...")
response = client.post('/api/analyze', data=json.dumps(payload))

print(f"\nStatus Code: {response.status_code}")
if response.status_code == 200:
    data = response.get_json()
    print("\n--- Pipeline Successfully Returned ---")
    print(f"Wellness Index: {data['wellness_index']}")
    print(f"Toxicity Score: {data['toxicity_score']}")
    print(f"Positivity Score: {data['positivity_score']}")
    print(f"Recommendation: {data['recommendation']}")
else:
    print(response.data)

