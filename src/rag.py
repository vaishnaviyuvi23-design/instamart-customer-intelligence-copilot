"""Part D tasks 1-3 - chunking, TF-IDF retrieval and a retrieval-grounded policy answerer (offline mock LLM)."""
from common import ROOT          # must come first: puts the repo root on sys.path
import mock_llm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICY_DIR = ROOT / "data" / "policies"
CHUNK_WORDS = 40

POLICY_SYSTEM = ("You are a POLICY ASSISTANT for Instamart support staff. Answer ONLY from the supplied CONTEXT. "
                 "If the context does not contain the answer, reply exactly: I could not find that information in the policy documents.")


def chunk_words(text, size=CHUNK_WORDS, overlap=10):
    """Windows of `size` words starting every `size-overlap` words; stop after the first window reaching the last word."""
    words, step, chunks, start = text.split(), size - overlap, [], 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            return chunks
        start += step


def load_chunks(overlap=10, size=CHUNK_WORDS):
    out = []
    for f in sorted(POLICY_DIR.glob("*.txt")):
        for c in chunk_words(f.read_text(encoding="utf-8"), size, overlap):
            out.append({"source": f.name, "text": c})
    return out


class Retriever:
    """TF-IDF index over policy chunks; the same fitted vectoriser transforms every query."""
    def __init__(self, overlap=10, size=CHUNK_WORDS):
        self.chunks = load_chunks(overlap, size)
        self.vec = TfidfVectorizer()
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix).ravel()
        order = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(sims[i]), 4), "text": self.chunks[i]["text"]} for i in order]

    def answer(self, question):
        hits = self.retrieve(question, k=3)
        context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        messages = [{"role": "system", "content": POLICY_SYSTEM},
                    {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
        resp = mock_llm.chat(messages)
        return {"answer": resp.choices[0].message.content, "sources": [h["source"] for h in hits]}


_default = Retriever(overlap=10)          # the index the agent's search_policy tool reuses


def retrieve(query, k=3):
    return _default.retrieve(query, k)


def answer(question):
    return _default.answer(question)
