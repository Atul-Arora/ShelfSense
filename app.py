# app.py
import streamlit as st
import requests
import os
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import time

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_CHAT_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"

if not MISTRAL_API_KEY:
    st.error("❌ Missing Mistral API key. Please set MISTRAL_API_KEY in your environment.")
    st.stop()

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="ShelfSense 📚", layout="wide", page_icon="📘")

# --- Styling ---
st.markdown(
    """
    <style>
      .stApp { background-color: #000000; color: #f5f5f5; }
      section[data-testid="stSidebar"] { background: #111111; color: #ffa500; }
      h1, h2, h3 { color: #ffa500; }
      .book-card { background-color:#1a1a1a; padding:16px; border-radius:12px; margin-bottom:20px; border: 1px solid #ffa50033; }
      .book-title { font-size:20px; font-weight:600; margin-bottom:6px; color:#ffa500; }
      .book-meta { margin:2px 0; font-size:14px; }
      .book-author { color:#ffcc66; }
      .book-category { color:#ff884d; }
      .book-desc { color:#e6e6e6; font-size:14px; line-height:1.5; margin-top:8px; }
      .stSuccess { background-color: #262626; border-left: 4px solid #ffa500; }
      .stInfo { background-color: #1a1a1a; border-left: 4px solid #ffcc66; }
      .stWarning { background-color: #331a00; border-left: 4px solid #ff884d; }
      .stError { background-color: #330000; border-left: 4px solid #ff3333; }
      .stTextInput > div > div > input {
          border: 2px solid orange !important;
          border-radius: 8px !important;
      }
      .stTextInput > div > div > input:focus {
          border: 2px solid #ff6600 !important;
          box-shadow: 0 0 8px #ff6600 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Helpers
# -------------------------
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

@st.cache_data
def load_books(csv_path="FINAL-database.csv"):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    title_col = find_col(df, ["Title", "title", "Book Title", "Book", "Name"])
    if not title_col:
        st.error("CSV must contain a Title column (e.g. 'Title' or 'Book Title').")
        st.stop()

    author_col = find_col(df, ["Author", "Authors", "author", "Writer"])
    category_col = find_col(df, ["Category", "Categories", "Genre", "Genres"])
    desc_col = find_col(df, ["Description", "description", "Desc", "Summary", "Synopsis"])

    df = df.rename(columns={title_col: "Title"})
    df["Author"] = df[author_col] if author_col else ""
    df["Category"] = df[category_col] if category_col else ""
    df["Description"] = df[desc_col] if desc_col else ""

    df = df.dropna(subset=["Title"]).reset_index(drop=True)
    df = df.fillna("")
    return df

books_df = load_books("FINAL-database.csv")

# -------------------------
# Embedding model
# -------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -------------------------
# FAISS Index
# -------------------------
@st.cache_resource
def build_local_index(df, _model, emb_file="book_embeddings_local.npy"):
    texts = (
        df["Title"].fillna("")
        + " " + df["Author"].fillna("")
        + " " + df["Category"].fillna("")
        + " " + df["Description"].fillna("")
    ).tolist()

    if os.path.exists(emb_file):
        embeddings = np.load(emb_file)
    else:
        embeddings = _model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms == 0, 1.0, norms)
        np.save(emb_file, embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings

faiss_index, book_embeddings = build_local_index(books_df, model)

# -------------------------
# Search
# -------------------------
def semantic_search(query, top_k=6, threshold=0.35):
    query_norm = query.strip().lower()
    results_list = []

    # Exact match
    exact = books_df[books_df["Title"].str.lower().str.strip() == query_norm]
    if not exact.empty:
        results_list.append(exact)

    # Partial match
    partial = books_df[books_df["Title"].str.lower().str.contains(query_norm, na=False)]
    if not partial.empty:
        results_list.append(partial)

    # Semantic search
    query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    qnorm = query_vec / np.linalg.norm(query_vec)
    distances, indices = faiss_index.search(qnorm, top_k)

    valid = [(idx, score) for idx, score in zip(indices[0], distances[0]) if score >= threshold]
    if valid:
        semantic = books_df.iloc[[idx for idx, _ in valid]]
        results_list.append(semantic)

    if results_list:
        combined = pd.concat(results_list).drop_duplicates(subset=["Title"]).reset_index(drop=True)
        return combined
    else:
        return pd.DataFrame()

# -------------------------
# Mistral API
# -------------------------
def ask_mistral(prompt, retries=3, backoff=5):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "mistral-small",
        "messages": [{"role": "user", "content": prompt}],
    }

    for attempt in range(retries):
        try:
            r = requests.post(MISTRAL_CHAT_ENDPOINT, headers=headers, json=data, timeout=30)
            if r.status_code == 429 and attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
                continue
            r.raise_for_status()
            j = r.json()
            return j["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
                continue
            return f"❌ Error contacting Mistral API: {e}"

def summarize_book(title):
    book = books_df[books_df["Title"].str.contains(title, case=False, na=False)]
    if book.empty:
        return "❌ Book not found."
    desc = book.iloc[0].get("Description", "")
    if not desc:
        return "❌ No description found to summarize."
    return ask_mistral(f"Summarize the following book description in 4 short sentences:\n\n{desc}")

# -------------------------
# UI
# -------------------------
st.sidebar.title("📚 ShelfSense")
st.sidebar.markdown("Your AI-powered library assistant.")
user_question = st.sidebar.text_input("Ask the Library Bot:")
if user_question:
    with st.spinner("🤖 Thinking..."):
        answer = ask_mistral(f"You are a helpful library assistant. User asked: {user_question}")
    st.sidebar.success(answer)

st.title("✨ Welcome to ShelfSense")
tabs = st.tabs(["🔍 Search Books", "📖 Summarize Book", "📔 Suggest by Category"])

# -------------------------
# TAB 1 - Search Books
# -------------------------
with tabs[0]:
    st.header("🔍 Search for a Book")
    search_query = st.text_input("Enter your search query:")
    if search_query:
        with st.spinner("🔎 Searching..."):
            results = semantic_search(search_query, top_k=6, threshold=0.35)

        if results.empty:
            st.error("❌ Book not found in your library.")
        else:
            cols = st.columns(1 if len(results) == 1 else 2)
            for i, (_, row) in enumerate(results.reset_index(drop=True).iterrows()):
                col = cols[i % len(cols)]
                with col:
                    st.markdown(
                        f"""
                        <div class="book-card">
                          <div class="book-title">{row['Title']}</div>
                          {'<div class="book-meta book-author">👤 ' + row['Author'] + '</div>' if row['Author'].strip() else ''}
                          {'<div class="book-meta book-category">📂 ' + row['Category'] + '</div>' if row['Category'].strip() else ''}
                          <div class="book-desc">{(row['Description'][:350] + '...') if len(row['Description']) > 350 else row['Description']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# -------------------------
# TAB 2 - Summarize Book
# -------------------------
with tabs[1]:
    st.header("📖 Summarize a Book")
    book_name = st.text_input("Enter book name to summarize:")
    if book_name:
        with st.spinner("📖 Summarizing..."):
            summary = summarize_book(book_name)
        st.success(summary)

# -------------------------
# TAB 3 - Suggest by Category
# -------------------------
with tabs[2]:
    st.header("📔 Suggest Books by Category")
    all_categories = []
    for cats in books_df["Category"].dropna().unique():
        all_categories.extend([c.strip() for c in str(cats).split(",") if c.strip()])
    all_categories = sorted(set(all_categories))

    selected = st.multiselect("📂 Choose categories:", options=all_categories)
    if selected:
        st.subheader(f"📚 Suggestions for: {', '.join(selected)}")
        filtered = books_df[
            books_df["Category"].apply(lambda x: any(cat.lower() in str(x).lower() for cat in selected))
        ]
        if not filtered.empty:
            for _, row in filtered.sample(min(8, len(filtered))).iterrows():
                st.markdown(
                    f"""
                    <div class="book-card">
                      <div class="book-title">{row['Title']}</div>
                      {'<div class="book-meta book-author">👤 ' + row['Author'] + '</div>' if row['Author'].strip() else ''}
                      <div class="book-desc">{(row['Description'][:250] + '...') if len(row['Description']) > 250 else row['Description']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No books found for the selected categories.")
    else:
        st.info("👉 Select one or more categories from the dropdown to see suggestions.")
