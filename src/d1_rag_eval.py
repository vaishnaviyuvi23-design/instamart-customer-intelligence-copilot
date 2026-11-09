"""Part D tasks 1-4 - chunk counts, retrieval check, grounded-answer evaluation with overlap 10 and overlap 0."""
from collections import Counter
import pandas as pd
from common import RAW
from rag import Retriever

ev = pd.read_csv(RAW / "policy_eval.csv", keep_default_na=False)

def evaluate(r, label):
    print(f"\n=== {label}: {len(r.chunks)} chunks {dict(Counter(c['source'] for c in r.chunks))}")
    doc_hit = fact_hit = fallback = 0
    n_ans = n_un = 0
    for q in ev.itertuples(index=False):
        out = r.answer(q.question)
        if q.expected_doc == "NONE":
            n_un += 1
            ok = out["answer"].startswith("I could not find")
            fallback += ok
            print(f"  [unanswerable] fallback={ok} | {q.question}")
        else:
            n_ans += 1
            d, f = q.expected_doc in out["sources"], q.expected_fact in out["answer"]
            doc_hit += d; fact_hit += f
            if not (d and f):
                print(f"  [FAIL] doc_in_top3={d} fact_in_answer={f} | {q.question}\n         answer: {out['answer']}")
    print(f"  answerable: doc in top-3 {doc_hit}/{n_ans}; fact in answer {fact_hit}/{n_ans}; unanswerable fallback {fallback}/{n_un}")

r10, r0 = Retriever(overlap=10), Retriever(overlap=0)
print("chunk counts, overlap 10:", len(r10.chunks), dict(Counter(c["source"] for c in r10.chunks)))
print("top-3 sources for 'refund policy late delivery':", [h["source"] for h in r10.retrieve("refund policy late delivery")])
evaluate(r10, "overlap 10")
evaluate(r0, "overlap 0")
