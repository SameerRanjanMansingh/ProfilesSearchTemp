import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [matches, setMatches] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [topK, setTopK] = useState(4);

  async function handleSearch(e: any) {
    e?.preventDefault();
    setLoading(true);
    setAnswer("");
    setMatches([]);
    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: topK }),
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || "Server error");
      }
      const data = await res.json();
      setMatches(data.matches || []);
      setAnswer(data.answer || "");
    } catch (err: any) {
      console.error(err);
      alert("Error: " + (err.message || err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", padding: 24 }}>
      <h1>RAG Profiles Search — Demo</h1>
      <p>Search profiles (local sample). Make sure backend is running and ingested.</p>

      <form onSubmit={handleSearch} style={{ marginTop: 12 }}>
        <input
          placeholder="e.g., frontend engineer experienced with React and Next.js"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ width: "60%", padding: 8, marginRight: 8 }}
        />
        <select value={topK} onChange={(e) => setTopK(Number(e.target.value))} style={{ marginRight: 8 }}>
          <option value={1}>1</option>
          <option value={2}>2</option>
          <option value={3}>3</option>
          <option value={4}>4</option>
        </select>
        <button disabled={loading} style={{ padding: "8px 12px" }}>{loading ? "Searching..." : "Search"}</button>
      </form>

      <section style={{ marginTop: 20 }}>
        <h2>Generated Answer</h2>
        <div style={{ whiteSpace: "pre-wrap", background: "#f6f6f6", padding: 12, borderRadius: 6 }}>
          {answer || "No answer yet."}
        </div>
      </section>

      <section style={{ marginTop: 20 }}>
        <h2>Matched Profiles</h2>
        {matches.length === 0 && <div>No matches.</div>}
        {matches.map((m, i) => (
          <div key={m.id || i} style={{ border: "1px solid #eee", padding: 12, marginTop: 8, borderRadius: 6 }}>
            <strong>{m.name}</strong> — {m.headline} <br />
            <small>{m.location} — {m.email || ""}</small>
            <p style={{ marginTop: 8 }}>{m.summary}</p>
            <p><em>Skills: {m.skills}</em></p>
          </div>
        ))}
      </section>
    </main>
  );
}