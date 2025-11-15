import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from supabase_integration import upsert_profiles_to_supabase

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = 384

def ingest_local_to_supabase(csv_path="profiles_sample.csv", normalize=True):
    df = pd.read_csv(csv_path)
    records = []
    docs = []
    for _, row in df.iterrows():
        text = f"Name: {row.get('name','')}\nHeadline: {row.get('headline','')}\nLocation: {row.get('location','')}\nSkills: {row.get('skills','')}\nSummary: {row.get('summary','')}"
        docs.append(text)
        # prepare metadata record; embedding filled below
        records.append({
            "id": str(row.get("id","")),
            "name": row.get("name",""),
            "headline": row.get("headline",""),
            "location": row.get("location",""),
            "skills": row.get("skills",""),
            "summary": row.get("summary",""),
            "email": row.get("email",""),
            # "embedding": will be set after computing embeddings
        })
    embeddings = embed_model.encode(docs, show_progress_bar=True, convert_to_numpy=True)
    if normalize:
        # normalize L2 to approximate cosine with Euclidean operator
        faiss.normalize_L2(embeddings)
    # attach embeddings to records and upsert
    for i, rec in enumerate(records):
        rec["embedding"] = embeddings[i].tolist()
    resp = upsert_profiles_to_supabase(records)
    return {"status": "ingested_to_supabase", "count": len(records), "resp": resp.data if hasattr(resp, "data") else resp}