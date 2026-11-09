"""Part D tasks 5-8 - refund tools, ReAct loop with safety limits, checkpointing and resume (offline mock LLM)."""
import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone

from common import DB, OUT, STATE         # must come first: puts the repo root on sys.path
import mock_llm
from rag import retrieve

LEDGER = OUT / "refund_ledger.jsonl"
RUNS = OUT / "agent_runs.jsonl"
LATE_MINUTES = 45
AUTO_LIMIT_INR = 1000

AGENT_SYSTEM = ("You are a refund AGENT for Instamart support. Look up the order, search the policies, calculate the refund, "
                "then issue it or escalate it to a human. Use the provided tools and finish with a one-line verdict "
                "starting REFUND, ESCALATE, DENIED or NOT_FOUND.")

_ctx = {"run_id": None}                  # lets the ledger tools tag their lines with the current run


# ------------------------------------------------------------------ tools
def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    dm = row[3]
    return {"order_id": row[0], "status": row[1], "order_amount": float(row[2]),
            "delivery_minutes": None if dm is None else (int(dm) if float(dm).is_integer() else float(dm))}


def search_policy(query):
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    """Full refund of the order amount if delivered in more than 45 minutes, otherwise 0."""
    if delivery_minutes is not None and delivery_minutes > LATE_MINUTES:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0}


def _ledger(tool, order_id, refund_inr):
    line = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": tool,
            "run_id": _ctx["run_id"], "order_id": order_id, "refund_inr": refund_inr}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(line) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


def _spec(name, description, props, required):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required}}

TOOLS_SPEC = [
    _spec("get_order", "Look up an order's status, amount and delivery time in the Instamart database.",
          {"order_id": {"type": "string", "description": "Order id such as ORD-100123"}}, ["order_id"]),
    _spec("search_policy", "Search the policy documents and return the three most relevant passages.",
          {"query": {"type": "string"}}, ["query"]),
    _spec("calculate_refund", "Compute the refund in INR: full amount if delivered in more than 45 minutes, else 0.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}, ["order_amount", "delivery_minutes"]),
    _spec("issue_refund", "Issue an automatic refund (up to INR 1000) and record it in the refund ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


# ------------------------------------------------------------------ loop
class SimulatedCrash(Exception):
    """Raised on purpose by the crash-and-resume demonstration."""


def _checkpoint(s):
    keys = ("run_id", "scenario_id", "messages", "tokens_used", "step", "tools_executed", "llm_calls", "steps",
            "max_iterations", "token_limit", "time_limit_seconds", "demo")
    (STATE / f"{s['run_id']}.json").write_text(json.dumps({k: s[k] for k in keys}, indent=1))


def _handoff(s, status, why):
    tools = ", ".join(x["tool"] for x in s["steps"]) or "none"
    return (f"HANDOFF to human agent ({status}): {why}. Scenario {s['scenario_id']}; {len(s['steps'])} steps completed "
            f"(tools: {tools}); tokens used {s['tokens_used']}. Please review the request manually.")


def _refund_amount(s):
    for st in reversed(s["steps"]):
        if st["tool"] in ("issue_refund", "escalate_to_human"):
            return st["observation"].get("refund_inr", 0)
    return 0


def _loop(s, crash_after_tools=None):
    t0 = time.monotonic()
    result = None
    while result is None:
        if time.monotonic() - t0 >= s["time_limit_seconds"]:                   # (c) wall-clock guard, top of iteration
            status = "FALLBACK_TIMEOUT"; result = _handoff(s, status, f"time limit of {s['time_limit_seconds']}s reached")
            break
        if s["llm_calls"] >= s["max_iterations"]:                              # (a) iteration guard
            status = "FALLBACK_MAX_ITERATIONS"; result = _handoff(s, status, f"no final answer after {s['max_iterations']} model calls")
            break
        resp = mock_llm.chat(s["messages"], tools=TOOLS_SPEC)
        s["llm_calls"] += 1
        s["tokens_used"] += resp.usage.total_tokens
        if s["tokens_used"] > s["token_limit"]:                                # (b) token guard: tool NOT executed
            status = "FALLBACK_TOKEN_LIMIT"; result = _handoff(s, status, f"token limit {s['token_limit']} exceeded ({s['tokens_used']})")
            break
        msg = resp.choices[0].message
        if not msg.tool_calls:
            result = msg.content
            status = result.split(":")[0].strip()
            break
        for tc in msg.tool_calls:
            s["messages"].append({"role": "assistant", "content": None,
                                  "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
            fn = TOOL_MAP.get(tc.name)
            try:
                obs = fn(**tc.arguments) if fn else {"error": f"unknown tool {tc.name}"}
            except Exception as e:                                             # a failing tool becomes an observation
                obs = {"error": f"{type(e).__name__}: {e}"}
            s["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
            s["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
            s["tools_executed"].append(tc.name)
            s["step"] += 1
            _checkpoint(s)                                                     # saved after every observation
            if crash_after_tools is not None and len(s["tools_executed"]) == crash_after_tools:
                raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools} ({tc.name})")
    out = {"run_id": s["run_id"], "scenario_id": s["scenario_id"], "status": status, "final_answer": result,
           "steps": s["steps"], "llm_calls": s["llm_calls"], "tokens_used": s["tokens_used"],
           "refund_inr": _refund_amount(s), "tools_executed": list(s["tools_executed"])}
    if s["demo"]:
        out["demo"] = s["demo"]
    with open(RUNS, "a") as f:
        f.write(json.dumps({k: out[k] for k in ("run_id", "scenario_id", "status", "final_answer", "llm_calls", "tokens_used",
                                                "refund_inr", "steps") + (("demo",) if s["demo"] else ())}) + "\n")
    return out


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None, demo=None):
    s = {"run_id": run_id or f"{scenario_id}-{uuid.uuid4().hex[:8]}", "scenario_id": scenario_id,
         "messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user_message}],
         "tokens_used": 0, "step": 0, "tools_executed": [], "llm_calls": 0, "steps": [],
         "max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds, "demo": demo}
    _ctx["run_id"] = s["run_id"]
    return _loop(s, crash_after_tools)


def resume_agent(run_id):
    """Reload state/<run_id>.json and continue the same run; tools already executed are never executed again."""
    s = json.loads((STATE / f"{run_id}.json").read_text())
    _ctx["run_id"] = run_id
    return _loop(s)
