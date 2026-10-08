"""Print recorded exchanges verbatim, as quoted in the paper. No model calls.

  python scripts/export_cases.py                      # the two cases of the paper
  python scripts/export_cases.py c_s2_partial_gptoss_B3ft B3ft u0096.r1:u0141
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
args = sys.argv[1:] or ["c_s2_partial_gptoss_B3ft", "B3ft", "u0096.r1:u0141", "u0096.r1:u0059"]
run, method, pairs = ROOT / "records" / args[0], args[1], [tuple(p.split(":")) for p in args[2:]]
data = ROOT / "data" / "community"
own = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in (data / "owner_private").glob("u*.json")}
pub = {json.loads(l)["person_id"]: json.loads(l) for l in (data / "public_directory" / "directory.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
latest = {}
for line in (run / "cases.jsonl").read_text(encoding="utf-8").splitlines():
    c = json.loads(line)
    k = (c["request_id"], c["candidate_id"])
    if c["method"] == method and k in pairs and (k not in latest or c.get("execution_status") == "ok"):
        latest[k] = c
for k in pairs:
    c = latest[k]; a, b = c["person_id"], c["candidate_id"]
    facts = {a: c.get("self_facts_a") or {}, b: c.get("self_facts_b") or {}}
    req = next(r["request_text"] for r in own[a]["requests"] if r["request_id"] == k[0])
    print(f"## {pub[a]['name']} ({a}) and {pub[b]['name']} ({b})")
    print(f"Request of {a}: {req}")
    print(f"Public profile of {b}: {pub[b]['public_card']}")
    print(f"Request of {b}: {own[b]['requests'][0]['request_text']}")
    for m in c["messages"]:
        if m["type"] == "ASK":
            print(f"{m['sender']} asks: {m['question']}")
        else:
            quote = ""
            if m.get("evidence_ids"):
                attr = m["evidence_ids"][0].split(".", 1)[1]
                quote = f' (quoting "{facts[m["sender"]].get(attr, {}).get("quote", "")}")'
            print(f"{m['sender']} answers: {m['answer']}{quote}")
    print(f"Decision: {a}->{b} {c['decision_ab']}, {b}->{a} {c['decision_ba']}, handshake {c['joint_decision']}\n")
