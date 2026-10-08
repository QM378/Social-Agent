"""Offline (no-LLM) matcher baselines computed from a run's pools + stored embeddings, appended to cases.jsonl.
Methods: tfidf_top3, rrf_top3 (BM25+embedding reciprocal rank fusion), graph_ppr_top3 (keyword bipartite graph, personalized
PageRank from the requester's request words), plus finalisation of llm_score_top3 (top-3 by stored LLM score within the pool).
All use ONLY public cards and the request text; never labels."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from social_agent.experiment.runner import BM25, _tokens, _cos, load_community_inputs

_STOP = set("the a an and or of to in on for with is are be i my me you your we our it its that this at as by from have has not no just really want looking someone partner session sessions like love would prefer need".split())


def _tfidf_vectors(docs: dict[str, str]):
    toks = {i: [t for t in _tokens(d) if t not in _STOP] for i, d in docs.items()}
    df = Counter(); n = len(docs)
    for v in toks.values():
        df.update(set(v))
    idf = {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}
    vec = {}
    for i, v in toks.items():
        tf = Counter(v); vec[i] = {w: tf[w] / len(v) * idf[w] for w in tf} if v else {}
    return vec, idf


def _sparse_cos(a: dict, b: dict) -> float:
    num = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values())); nb = math.sqrt(sum(v * v for v in b.values()))
    return num / (na * nb) if na and nb else 0.0


def _ppr(edges: dict[str, dict[str, float]], seeds: dict[str, float], alpha: float = 0.15, iters: int = 30) -> dict[str, float]:
    nodes = set(edges) | {v for nb in edges.values() for v in nb}
    s = sum(seeds.values()) or 1.0; p0 = {n: seeds.get(n, 0.0) / s for n in nodes}; p = dict(p0)
    for _ in range(iters):
        nxt = {n: alpha * p0[n] for n in nodes}
        for u, nb in edges.items():
            if not nb or p[u] == 0:
                continue
            tot = sum(nb.values())
            for v, w in nb.items():
                nxt[v] += (1 - alpha) * p[u] * w / tot
        p = nxt
    return p


def run_offline_baselines(cdir: Path, run_dir: Path, methods: list[str], view: str = "partial") -> dict[str, Any]:
    inputs = load_community_inputs(cdir, view)
    pub = {pid: d["public_text"] for pid, d in inputs["public_directory"].items()}
    pools = json.loads((run_dir / "pools.json").read_text(encoding="utf-8"))["pools"]
    vec_path = run_dir / "embeddings.json"
    vec = json.loads(vec_path.read_text(encoding="utf-8")) if vec_path.exists() else {"cards": {}, "requests": {}}
    personas = inputs["personas"]
    qtext = {}
    for r in inputs["requests"]:
        p = personas[r["person_id"]]
        qtext[r["request_id"]] = p["request_text"] if r["kind"] == "primary" else next(a["request_text"] for a in p["alt_requests"] if a["request_id"] == r["request_id"])
    tfv, idf = _tfidf_vectors(pub)
    # keyword bipartite graph: person <-> word (public card), weights = tfidf
    edges: dict[str, dict[str, float]] = defaultdict(dict)
    for pid, v in tfv.items():
        for w, x in v.items():
            edges[pid][f"w:{w}"] = x; edges[f"w:{w}"][pid] = x
    by_scen: dict[str, list[str]] = defaultdict(list)
    for pid, p in personas.items():
        by_scen[p["scenario"]].append(pid)
    bm = {sc: BM25({pid: pub[pid] for pid in ids}) for sc, ids in by_scen.items()}
    # existing cases: which (rid, j, method) already present; llm_score records to finalise
    cases_path = run_dir / "cases.jsonl"; existing = set(); score_recs: dict[str, dict[str, float]] = defaultdict(dict)
    if cases_path.exists():
        for line in cases_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line); existing.add((d["request_id"], d["candidate_id"], d["method"]))
            if d["method"] == "llm_score_top3" and d.get("llm_score") is not None:
                score_recs[d["request_id"]][d["candidate_id"]] = d["llm_score"]
    prim_pool = {v["person_id"]: v["pool"] for k, v in pools.items() if k.endswith(".r1")}
    n = Counter()
    with open(cases_path, "a", encoding="utf-8") as f:
        for rid, pl in pools.items():
            pid = pl["person_id"]; kind = "primary" if rid.endswith(".r1") else "alternative"
            pool = pl["pool"] if kind == "primary" else prim_pool.get(pid, pl["pool"])
            if not pool:
                continue
            sc = personas[pid]["scenario"]
            rankings: dict[str, list[str]] = {}
            if "tfidf_top3" in methods:
                qv, _ = _tfidf_vectors({"q": qtext[rid]}); q = qv["q"]
                rankings["tfidf_top3"] = sorted(pool, key=lambda j: -_sparse_cos(q, tfv[j]))
            if "rrf_top3" in methods:
                bmr = sorted(pool, key=lambda j: -bm[sc].score(qtext[rid], j))
                emr = sorted(pool, key=lambda j: -_cos(vec["requests"].get(rid, []), vec["cards"].get(j, []))) if vec["requests"].get(rid) else bmr
                rr = {j: 1 / (60 + bmr.index(j) + 1) + 1 / (60 + emr.index(j) + 1) for j in pool}
                rankings["rrf_top3"] = sorted(pool, key=lambda j: -rr[j])
            if "graph_ppr_top3" in methods:
                qv, _ = _tfidf_vectors({"q": qtext[rid]}); seeds = {f"w:{w}": x for w, x in qv["q"].items() if f"w:{w}" in edges}
                pr = _ppr(edges, seeds or {pid: 1.0})
                rankings["graph_ppr_top3"] = sorted(pool, key=lambda j: -pr.get(j, 0.0))
            if "llm_score_top3" in methods and score_recs.get(rid):
                rankings["llm_score_top3"] = sorted(pool, key=lambda j: -(score_recs[rid].get(j) or -1))
            for m, ranked in rankings.items():
                for j in pool:
                    if (rid, j, m) in existing and m != "llm_score_top3":
                        continue
                    rec = {"request_id": rid, "person_id": pid, "candidate_id": j, "method": m, "view": view, "request_kind": kind,
                           "execution_status": "ok", "error": None, "rank": ranked.index(j) + 1,
                           "decision_ab": "recommend" if j in ranked[:3] else "insufficient_info",
                           "usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0},
                           "finished_utc": datetime.now(timezone.utc).isoformat(), "offline": True}
                    if m == "llm_score_top3":
                        rec["llm_score"] = score_recs[rid].get(j)
                    f.write(json.dumps(rec) + "\n"); n[m] += 1
    return {"appended": dict(n), "note": "top-3 within the pool = recommend, others insufficient_info (rank-based baselines make no reject)"}
