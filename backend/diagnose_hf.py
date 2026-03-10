"""Diagnose HF API calls — shows exactly what each model returns."""
import requests
import base64
import io
import json
from PIL import Image

def img_to_bytes(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        if max(img.size) > 800:
            img.thumbnail((800, 800))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return buf.getvalue()

HF_API = "https://api-inference.huggingface.co/models"
test_img = "test_images\\1000420307.jpg"
img_bytes = img_to_bytes(test_img)
b64_img = base64.b64encode(img_bytes).decode("utf-8")

print(f"Image: {test_img} ({len(img_bytes)} bytes)\n")

# 1. BLIP Image Captioning
print("=" * 50)
print("1. BLIP Image Captioning")
try:
    r = requests.post(
        f"{HF_API}/Salesforce/blip-image-captioning-base",
        data=img_bytes,
        headers={"Content-Type": "application/octet-stream"},
        timeout=60
    )
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.text[:500]}")
except Exception as e:
    print(f"   ERROR: {e}")

# 2. CLIP Zero-shot Classification
print("\n" + "=" * 50)
print("2. CLIP Zero-shot Image Classification")
labels = ["video", "reel", "static image", "text post", "meme", "news article"]
try:
    r = requests.post(
        f"{HF_API}/openai/clip-vit-base-patch32",
        json={
            "inputs": b64_img,
            "parameters": {"candidate_labels": labels}
        },
        timeout=60
    )
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.text[:500]}")
except Exception as e:
    print(f"   ERROR: {e}")

# 2b. Try CLIP with different endpoint format
print("\n" + "=" * 50)
print("2b. CLIP via zero-shot-image-classification task")
try:
    r = requests.post(
        f"{HF_API}/openai/clip-vit-base-patch32",
        data=img_bytes,
        headers={"Content-Type": "application/octet-stream"},
        timeout=60
    )
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.text[:500]}")
except Exception as e:
    print(f"   ERROR: {e}")

# 3. Toxic-bert (with sample text)
print("\n" + "=" * 50)
print("3. Toxic-bert (text classification)")
test_text = "a scary horror scene with blood and violence"
try:
    r = requests.post(
        f"{HF_API}/unitary/toxic-bert",
        json={"inputs": test_text},
        timeout=60
    )
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.text[:500]}")
except Exception as e:
    print(f"   ERROR: {e}")

# 4. Sentiment (with sample text)
print("\n" + "=" * 50)
print("4. RoBERTa Sentiment (text classification)")
try:
    r = requests.post(
        f"{HF_API}/cardiffnlp/twitter-roberta-base-sentiment-latest",
        json={"inputs": test_text},
        timeout=60
    )
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.text[:500]}")
except Exception as e:
    print(f"   ERROR: {e}")

# 5. Text embedding
print("\n" + "=" * 50)
print("5. Sentence-transformers (feature extraction)")
try:
    r = requests.post(
        f"{HF_API}/sentence-transformers/all-MiniLM-L6-v2",
        json={"inputs": test_text},
        timeout=60
    )
    print(f"   Status: {r.status_code}")
    print(f"   Response length: {len(r.text)} chars")
    data = r.json()
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], list):
            print(f"   Embedding dim: {len(data[0])}")
        else:
            print(f"   Result type: {type(data[0])}")
except Exception as e:
    print(f"   ERROR: {e}")
