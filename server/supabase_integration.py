from supabase import create_client
import os
import numpy as np

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Set SUPABASE_URL and SUPABASE_KEY environment variables")

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

def upsert_profiles_to_supabase(records):
    """
    records: list of dicts like:
      {
        "id": "1",
        "name": "Alice",
        "headline": "...",
        ...,
        "embedding": [float, float, ...]  # length 384
      }
    Uses upsert so you can re-run ingestion
    """
    # supabase.table("profiles").upsert accepts list of dicts
    res = sb.table("profiles").upsert(records).execute()
    return res

def search_profiles_supabase(query_embedding, k=4):
    """
    query_embedding: np.ndarray or list (length 384). IMPORTANT: If you stored normalized vectors,
    normalize this query similarly before passing it.
    Returns rows from the match_profiles RPC.
    """
    if isinstance(query_embedding, np.ndarray):
        q = query_embedding.tolist()
    else:
        q = query_embedding

    # Call the SQL function defined earlier
    # The RPC param name must match the SQL function signature: match_profiles(query, k)
    res = sb.rpc("match_profiles", {"query": q, "k": k}).execute()
    if res.status_code not in (200, 201):
        # fall back to raw SQL via supabase.postgrest if needed
        raise RuntimeError(f"Supabase RPC error: {res.data} status {res.status_code}")
    return res.data