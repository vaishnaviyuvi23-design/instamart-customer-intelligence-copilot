"""Part D tasks 5-8 - run the 7 scenarios, the two stop-safely demos and the crash-and-resume demo."""
import pandas as pd
from common import RAW, STATE
import agent
from agent import RUNS, LEDGER, SimulatedCrash, resume_agent, run_agent

for p in (RUNS, LEDGER, *STATE.glob("*.json")):          # fresh, reproducible artefacts for every run
    p.unlink(missing_ok=True)

sc = pd.read_csv(RAW / "agent_scenarios.csv")
rows = []
for r in sc.itertuples(index=False):
    out = run_agent(r.message, r.scenario_id, run_id=f"{r.scenario_id}-main")
    expected = r.expected_outcome
    ok = (out["status"].startswith("FALLBACK") if expected == "FALLBACK" else out["status"] == expected) and \
         abs(out["refund_inr"] - float(r.expected_refund_inr)) < 0.01
    rows.append({"scenario": r.scenario_id, "status": out["status"], "refund_inr": out["refund_inr"],
                 "model_calls": out["llm_calls"], "tokens_used": out["tokens_used"],
                 "expected": expected, "expected_refund": r.expected_refund_inr, "match": ok})
tbl = pd.DataFrame(rows)
print(tbl.to_string(index=False))
assert tbl["match"].all(), "a scenario deviates from the reference"
s1 = sc.iloc[0]

print("\n--- stop-safely demos (scenario S1) ---")
d1 = run_agent(s1.message, "S1", token_limit=200, run_id="S1-token200", demo="token_limit=200")
print("token_limit=200 ->", d1["status"], "| calls", d1["llm_calls"], "| tokens", d1["tokens_used"], "| steps", len(d1["steps"]))
print("   final_answer:", d1["final_answer"])
d2 = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1-timeout0", demo="time_limit_seconds=0")
print("time_limit_seconds=0 ->", d2["status"], "| calls", d2["llm_calls"], "| steps", len(d2["steps"]))
print("   final_answer:", d2["final_answer"])

print("\n--- crash and resume (scenario S1) ---")
try:
    run_agent(s1.message, "S1", run_id="S1-crash", crash_after_tools=2, demo="crash-after-2nd-tool")
except SimulatedCrash as e:
    print("crashed:", e)
cp = __import__("json").loads((STATE / "S1-crash.json").read_text())
print("checkpoint on disk -> tools_executed:", cp["tools_executed"], "| step:", cp["step"], "| tokens_used:", cp["tokens_used"])
res = resume_agent("S1-crash")
print("resumed ->", res["status"], res["refund_inr"], "| llm_calls", res["llm_calls"])
print("tools executed across crash + resume:", res["tools_executed"])
assert res["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
assert res["status"] == "REFUND" and abs(res["refund_inr"] - 128.25) < 0.01
print("\nagent_runs.jsonl lines:", sum(1 for _ in open(RUNS)), "| ledger lines:", sum(1 for _ in open(LEDGER)),
      "| checkpoints:", len(list(STATE.glob('*.json'))))
