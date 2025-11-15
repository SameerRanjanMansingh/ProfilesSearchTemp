-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Profiles table with a 384-dim vector (all-MiniLM-L6-v2 -> 384)
CREATE TABLE IF NOT EXISTS profiles (
  id TEXT PRIMARY KEY,
  name TEXT,
  headline TEXT,
  location TEXT,
  skills TEXT,
  summary TEXT,
  email TEXT,
  embedding vector(384)
);

-- Optional helper function to return nearest neighbors (Euclidean / use normalized vectors for cosine)
CREATE OR REPLACE FUNCTION match_profiles(query vector, k integer)
RETURNS TABLE (
  id text,
  name text,
  headline text,
  location text,
  skills text,
  summary text,
  email text,
  distance float
) AS $$
  SELECT id,name,headline,location,skills,summary,email, (embedding <-> query) AS distance
  FROM profiles
  ORDER BY distance
  LIMIT k;
$$ LANGUAGE sql STABLE;