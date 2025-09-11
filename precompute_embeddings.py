import pandas as pd
import numpy as np
import requests, os, time
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Setup ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
EMBED_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent"

# Session with retry for SSL / network issues
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=2,  # exponential backoff (2s, 4s, 8s...)
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["POST"]
)
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)

def get_embedding(text):
    """Get embedding with retries & rate limiting"""
    headers = {"Content-Type": "application/json"}
    params = {"key": API_KEY}
    data = {"model": "models/embedding-001", "content": {"parts": [{"text": text}]}}

    try:
        response = session.post(EMBED_ENDPOINT, headers=headers, params=params, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return np.array(result["embedding"]["value"], dtype=np.float32)
    except Exception as e:
        print(f"❌ Failed to embed text ({e}), returning zeros")
        return np.zeros(768, dtype=np.float32)

# --- Load books ---
df = pd.read_csv("Books.csv - Sheet1.csv")
df = df.dropna(subset=["Title"])

texts = (
    df["Title"].fillna("") + " " +
    df["Author"].fillna("") + " " +
    df["Keywords"].fillna("") + " " +
    df["Description"].fillna("")
).tolist()

# --- Resume logic ---
output_file = "book_embeddings.npy"
embeddings = []

if os.path.exists(output_file):
    embeddings = list(np.load(output_file))
    print(f"🔄 Resuming from {len(embeddings)} embeddings...")

# --- Embed with safe rate limiting ---
for i, t in enumerate(texts[len(embeddings):], start=len(embeddings) + 1):
    vec = get_embedding(t)
    embeddings.append(vec)
    print(f"✅ Embedded {i}/{len(texts)}")

    # Save after each embedding
    np.save(output_file, np.vstack(embeddings))

    # Prevent rate limit (sleep 2s between calls)
    time.sleep(2)

print("🎉 All embeddings saved to book_embeddings.npy")
