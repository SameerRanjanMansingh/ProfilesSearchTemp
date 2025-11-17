from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from supabase_integration import search_profiles_supabase
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# generator as before
gen_tokenizer = AutoTokenizer.from_pretrained("t5-small")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
generator = pipeline("text2text-generation", model=gen_model, tokenizer=gen_tokenizer, device=-1)

def query_supabase_and_generate(query_text, top_k=4):
    q_emb = embed_model.encode([query_text], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)  # if you store normalized vectors
    q_emb = q_emb[0]
    rows = search_profiles_supabase(q_emb, k=top_k)
    # rows is a list of dicts with profile fields and distance
    matches = rows  # adapt if your RPC returns different keys
    # build context prompt same as before
    context = "\n\n".join([
        f"Profile {i+1}:\nName: {m.get('name')}\nHeadline: {m.get('headline','')}\nLocation: {m.get('location','')}\nSkills: {m.get('skills','')}\nSummary: {m.get('summary','')}"
        for i, m in enumerate(matches)
    ])
    prompt = f"You are an assistant helping to find candidate profiles.\n\nContext:\n{context}\n\nQuestion: {query_text}\n\nAnswer with a concise summary and list the most relevant profile(s) by Name and why they fit."
    generated = generator(prompt, max_length=256, do_sample=False)
    return {"matches": matches, "answer": generated[0]["generated_text"]}