"""Part C - support-ticket triage with guardrails: redaction, input limits, retries, output validation,
token budget, zero-shot vs few-shot comparison and segment-aware routing."""
import json
import random
import re
import time

import pandas as pd

from common import OUT, RAW   # must come first: it puts the repo root on sys.path
import mock_llm
from mock_llm import TransientLLMError, est_tokens

CATEGORIES = ["billing", "delivery", "technical", "fraud"]
MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
random.seed(42)                                     # reproducible jitter

# ------------------------------------------------------------------ redaction (order matters: email, phone, order id)
PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

def redact(text, counts=None):
    for name, pat, token in PATTERNS:
        text, n = pat.subn(token, text)
        if counts is not None:
            counts[name] = counts.get(name, 0) + n
    return text

def pii_left(text):
    return sum(len(pat.findall(text)) for _, pat, _ in PATTERNS)

# ------------------------------------------------------------------ prompts
ZERO_SHOT = ("You CLASSIFY a customer support ticket into exactly one category: billing, delivery, technical or fraud. "
             'Reply with JSON only, like {"category": "...", "reason": "..."}.')
FEW_SHOT = ("You CLASSIFY a customer support ticket into exactly one category: billing, delivery, technical or fraud. "
            'Reply with JSON only, like {"category": "...", "reason": "..."}.\n'
            "EXAMPLES:\n"
            'Ticket: Someone hacked my account and I did not make this unauthorized transaction. -> {"category": "fraud", "reason": "unauthorized access"}\n'
            'Ticket: The app freezes and shows an error when I open the cart. -> {"category": "technical", "reason": "app fault"}\n'
            'Ticket: I was charged twice and the bill is wrong. -> {"category": "billing", "reason": "wrong charge"}\n'
            'Ticket: My order is late and the rider is not answering. -> {"category": "delivery", "reason": "late rider"}')

# ------------------------------------------------------------------ resilient call
RETRY_LOG = []

def call_with_retry(messages, ticket_id, stats):
    """Call the model; on TransientLLMError wait BASE_DELAY*2**n*jitter and retry, at most MAX_RETRIES times."""
    for m in messages:                                          # hard privacy guard: nothing sensitive may leave
        assert pii_left(m["content"]) == 0, f"PII in prompt for {ticket_id}"
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            stats["transient_errors"] += 1
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            RETRY_LOG.append({"ticket_id": ticket_id, "retry": n, "wait": wait})
            print(f"  [retry] ticket={ticket_id} retry={n} wait={wait:.4f}s")
            time.sleep(wait)

# ------------------------------------------------------------------ validation
def validate(reply):
    """Return (status, category). Never guess: anything unclear is flagged for human review."""
    try:
        data = json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (reply or "").lower()))
        found = [c for c in CATEGORIES if c in words]
        return ("classified_keyword", found[0]) if len(found) == 1 else ("flagged_invalid_output", "")
    if isinstance(data, dict) and data.get("category") in CATEGORIES:
        return "classified_json", data["category"]
    return "flagged_invalid_output", ""

# ------------------------------------------------------------------ pipeline
def run_pipeline(system_prompt, label, token_budget=None, quiet_retries=True):
    tickets = pd.read_csv(RAW / "tickets.csv", dtype=str, keep_default_na=False)
    mock_llm.reset()                                            # fresh failure counters for each full run
    RETRY_LOG.clear()
    stats = {"transient_errors": 0}
    running, rows = 0, []
    print(f"\n--- run: {label} (budget={token_budget}) ---")
    for t in tickets.itertuples(index=False):
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "status": "", "category": "", "tokens_used": 0,
               "true_category": t.true_category}
        text = t.text
        if not text.strip():
            row["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif token_budget is not None and running >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": redact(text)}]
            try:
                resp = call_with_retry(messages, t.ticket_id, stats)
            except TransientLLMError:
                row["status"] = "permanent_error"
            else:
                row["tokens_used"] = resp.usage.total_tokens
                running += resp.usage.total_tokens
                row["status"], row["category"] = validate(resp.choices[0].message.content)
        rows.append(row)
    df = pd.DataFrame(rows)
    stats["retry_waits_ok"] = all(BASE_DELAY * 2 ** r["retry"] * 0.9 <= r["wait"] <= BASE_DELAY * 2 ** r["retry"] * 1.15
                                  for r in RETRY_LOG)
    stats["retry_lines"] = len(RETRY_LOG)
    return df, stats

def summarize(df, stats, label):
    cls = df[df["status"].isin(["classified_json", "classified_keyword"])]
    acc = (cls["category"] == cls["true_category"]).mean() if len(cls) else float("nan")
    s = {"run": label, "classified": len(cls), "correct": int((cls["category"] == cls["true_category"]).sum()),
         "accuracy_pct": round(float(acc) * 100, 1), "total_tokens": int(df["tokens_used"].sum()),
         "tokens_per_classified": round(float(df["tokens_used"].sum()) / max(len(cls), 1), 1)}
    print(f"status counts: {df['status'].value_counts().to_dict()}")
    print(f"transient errors={stats['transient_errors']} retry log lines={stats['retry_lines']} "
          f"waits within jitter range={stats['retry_waits_ok']}")
    print(s)
    return s

def route(df):
    seg = pd.read_csv(OUT / "customer_segments.csv")
    seg = seg.drop_duplicates("customer_id").set_index("customer_id")
    df = df.copy()
    df["segment_name"] = df["customer_id"].map(seg["segment_name"]).fillna("")
    prem = df["customer_id"].map(seg["is_premium"]).fillna(False).astype(bool)
    human = df["status"].isin(["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted", "permanent_error"])
    fraud = ~human & (df["category"] == "fraud")
    premium = ~human & ~fraud & prem
    df["route"] = "support_queue"
    df.loc[premium, "route"] = "premium_desk"
    df.loc[fraud, "route"] = "security_team"
    df.loc[human, "route"] = "human_review"
    df["priority"] = "NORMAL"
    df.loc[fraud | premium, "priority"] = "HIGH"
    return df[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]]

if __name__ == "__main__":
    tk = pd.read_csv(RAW / "tickets.csv", dtype=str, keep_default_na=False)
    print("empty tickets:", int((tk["text"].str.strip() == "").sum()),
          "| long tickets:", int((tk["text"].map(est_tokens) > MAX_INPUT_TOKENS).sum()),
          "| est_tokens of a long ticket:", int(tk["text"].map(est_tokens).max()))

    counts = {}
    redacted = [redact(x, counts) for x in tk["text"]]
    print("substitutions:", counts, "total:", sum(counts.values()))
    left = sum(pii_left(x) for x in redacted)
    print("PII matches left after re-applying the three patterns:", left); assert left == 0
    print("system prompt est tokens: zero-shot", est_tokens(ZERO_SHOT), "| few-shot", est_tokens(FEW_SHOT))
    assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT and "fraud" in FEW_SHOT

    zs, zs_stats = run_pipeline(ZERO_SHOT, "zero-shot")
    s_zs = summarize(zs, zs_stats, "zero-shot")
    cls = zs[zs["status"].isin(["classified_json", "classified_keyword"])]
    print("zero-shot confusion (rows=true, cols=predicted):")
    print(pd.crosstab(cls["true_category"], cls["category"]).to_string())

    fs, fs_stats = run_pipeline(FEW_SHOT, "few-shot")
    s_fs = summarize(fs, fs_stats, "few-shot")
    print("\ncomparison:"); print(pd.DataFrame([s_zs, s_fs]).to_string(index=False))

    demo_budget = 6000
    demo, d_stats = run_pipeline(ZERO_SHOT, "zero-shot budget demo", token_budget=demo_budget)
    summarize(demo, d_stats, "budget demo")
    route(demo).to_csv(OUT / "triage_budget_demo.csv", index=False)
    print(f"budget_exhausted tickets at TOKEN_BUDGET={demo_budget}: {int((demo['status'] == 'budget_exhausted').sum())}")

    final, f_stats = run_pipeline(FEW_SHOT, "final few-shot", token_budget=100000)
    summarize(final, f_stats, "final")
    res = route(final)
    res.to_csv(OUT / "triage_results.csv", index=False)
    print("\nfinal rows:", len(res))
    print(res.groupby(["route", "priority"]).size().to_string())
    with open(OUT / "triage_retry_log.txt", "w") as f:                # retry lines of the final run
        for r in RETRY_LOG:
            f.write(f"ticket={r['ticket_id']} retry={r['retry']} wait={r['wait']:.4f}s\n")
