"""
Improved FastAPI server for Profiles RAG demo.

- Uses absolute path (relative to this file) to find profiles_sample.csv.
- Adds /health and /index_info endpoints.
- Better error messages for missing CSV and missing Supabase table.
- Lazy-loads heavy models to keep startup fast.
"""
import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "profiles_sample.csv")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.pkl")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-small"
EMBED_DIM = 384

app = FastAPI(title="Profiles RAG API (improved)")

# CORS for local frontend

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# lazy globals
embed_model = None
generator = None
faiss_index = None
metadata = []

def ensure_embedding_model():
    global embed_model
    if embed_model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model:", EMBED_MODEL_NAME)
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        print("Embedding model loaded.")

def ensure_generator():
    global generator
    if generator is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        print("Loading generation model:", GEN_MODEL_NAME)
        gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
        gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
        generator = pipeline("text2text-generation", model=gen_model, tokenizer=gen_tokenizer, device=-1)
        print("Generation model loaded.")

def save_index():
    global faiss_index, metadata
    if faiss_index is None:
        return
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(faiss_index, f)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

def load_index():
    global faiss_index, metadata
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        with open(INDEX_PATH, "rb") as f:
            faiss_index = pickle.load(f)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        return True
    return False

@app.on_event("startup")
def startup():
    loaded = load_index()
    if loaded:
        print("Loaded FAISS index and metadata (from disk).")
    else:
        print("No index on startup. Call /ingest_local or /ingest_supabase to create one.")

class QueryPayload(BaseModel):
    query: str
    top_k: int = 4

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/index_info")
def index_info():
    global faiss_index, metadata, embed_model, generator
    return {
        "index_built": faiss_index is not None,
        "num_items": int(faiss_index.ntotal) if faiss_index is not None else 0,
        "embedding_loaded": embed_model is not None,
        "generator_loaded": generator is not None,
    }

@app.post("/ingest_local")
def ingest_local():
    """
    Load sample profiles from server/profiles_sample.csv and index them.
    Uses absolute CSV_PATH located next to this file.
    """
    global faiss_index, metadata
    # ensure file exists
    if not os.path.exists(CSV_PATH):
        raise HTTPException(
            status_code=500,
            detail=f"profiles_sample.csv not found at {CSV_PATH}. Make sure the file exists."
        )

    # lazy load embedding model
    ensure_embedding_model()
    import pandas as pd
    import faiss
    import numpy as np

    df = pd.read_csv(CSV_PATH)
    docs = []
    metadata = []
    for _, row in df.iterrows():
        text = (
            f"Name: {row.get('name','')}\n"
            f"Headline: {row.get('headline','')}\n"
            f"Location: {row.get('location','')}\n"
            f"Skills: {row.get('skills','')}\n"
            f"Summary: {row.get('summary','')}\n"
        )
        docs.append(text)
        metadata.append({
            "id": str(row.get("id","")),
            "name": row.get("name",""),
            "headline": row.get("headline",""),
            "location": row.get("location",""),
            "skills": row.get("skills",""),
            "summary": row.get("summary",""),
            "email": row.get("email","")
        })

    embeddings = embed_model.encode(docs, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    faiss_index = index
    save_index()
    return {"status": "ingested", "count": len(docs), "csv_path": CSV_PATH}

@app.post("/ingest_supabase")
def ingest_supabase():
    """
    Example scaffold for fetching profiles from Supabase.
    Expects SUPABASE_URL and SUPABASE_KEY in env.
    """
    from supabase import create_client
    import pandas as pd
    import faiss
    import numpy as np

    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(status_code=400, detail="Set SUPABASE_URL and SUPABASE_KEY env variables before calling this endpoint.")

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # try to fetch rows; handle table-not-found errors gracefully
    try:
        result = sb.table("profiles").select("*").execute()
    except Exception as e:
        # likely table not found or permission error
        raise HTTPException(
            status_code=500,
            detail=("Could not read table 'profiles' from Supabase. "
                    "Make sure the table exists and SUPABASE_KEY has permission. "
                    f"Original error: {repr(e)}")
        )

    rows = result.data
    if not rows:
        raise HTTPException(status_code=404, detail="No rows returned from 'profiles' table.")

    # ensure embedding model
    ensure_embedding_model()
    docs = []
    metadata = []
    for row in rows:
        text = (
            f"Name: {row.get('name','')}\n"
            f"Headline: {row.get('headline','')}\n"
            f"Location: {row.get('location','')}\n"
            f"Skills: {row.get('skills','')}\n"
            f"Summary: {row.get('summary','')}\n"
        )
        docs.append(text)
        metadata.append({
            "id": str(row.get("id","")),
            "name": row.get("name",""),
            "headline": row.get("headline",""),
            "location": row.get("location",""),
            "skills": row.get("skills",""),
            "summary": row.get("summary",""),
            "email": row.get("email","")
        })

    embeddings = embed_model.encode(docs, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    global faiss_index
    faiss_index = index
    save_index()
    return {"status": "ingested_supabase", "count": len(docs)}

@app.post("/query")
def query(payload: QueryPayload):
    global faiss_index, metadata
    if faiss_index is None:
        raise HTTPException(status_code=400, detail="Index not built. Call /ingest_local first.")
    ensure_embedding_model()
    ensure_generator()
    import faiss
    import numpy as np

    q_emb = embed_model.encode([payload.query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    k = min(payload.top_k, faiss_index.ntotal)
    if k <= 0:
        return {"matches": [], "answer": ""}

    D, I = faiss_index.search(q_emb, k)
    indices = I[0].tolist()
    matches = [metadata[i] for i in indices]
    context = "\n\n".join([
        f"Profile {i+1}:\nName: {m.get('name')}\nHeadline: {m.get('headline','')}\nLocation: {m.get('location','')}\nSkills: {m.get('skills','')}\nSummary: {m.get('summary','')}"
        for i, m in enumerate(matches)
    ])
    prompt = (
        "You are an assistant helping to find candidate profiles.\n\n"
        f"Context:\n{context}\n\nQuestion: {payload.query}\n\n"
        "Answer with a concise summary and list the most relevant profile(s) by Name and why they fit."
    )
    generated = generator(prompt, max_length=256, do_sample=False)
    answer = generated[0]["generated_text"]
    return {"matches": matches, "answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)