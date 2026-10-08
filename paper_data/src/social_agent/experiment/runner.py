"""The per-pair run loop of the experiments: candidate lists, the LLM baselines, and the method (protocol.run_pair).

Agents and judges see the other person's public profile only. Reference answers (<community>/evaluator/) are never read
here; scoring reads them separately. Each run writes run_meta.json and cases.jsonl in its run folder and appends one line
to registry.jsonl next to it.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field

from social_agent.agent import PROMPT_VERSION, VOCAB_DESCRIPTIONS, vocab_block
from social_agent.vocab import load_vocab
from social_agent.llm import CallBudget, LLMConfig, make_client
from social_agent.protocol import METHODS, run_pair

EXP_VERSION = "community_exp_v1"


# ----------------------------------------------------------------------------- inputs (no labels)

def load_community_inputs(cdir: Path, view: str = "partial") -> dict[str, Any]:
    """agent_inputs-shaped dict from public_directory/ + owner_private/ (never evaluator/)."""
    directory = {}
    for line in (cdir / "public_directory" / "directory.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line); directory[d["person_id"]] = d
    personas = {}
    requests = []
    for f in sorted((cdir / "owner_private").glob("u*.json")):
        o = json.loads(f.read_text(encoding="utf-8")); pid = o["person_id"]
        prim = next(r for r in o["requests"] if r["kind"] == "primary")
        alts = [r for r in o["requests"] if r["kind"] == "alternative"]
        personas[pid] = {"persona_id": pid, "name": o["name"], "scenario": o["interaction_type"].replace("_partner", ""),
                         "self_description": o["self_text"] or "", "request_text": prim["request_text"] or "",
                         "request_id": prim["request_id"], "alt_requests": [{"request_id": a["request_id"], "request_text": a["request_text"] or ""} for a in alts],
                         "evidence": [], "disclosure_policy": o["disclosure_policy"], "generation_status": o["generation_status"]}
        requests.append({"request_id": prim["request_id"], "person_id": pid, "kind": "primary"})
        for a in alts:
            requests.append({"request_id": a["request_id"], "person_id": pid, "kind": "alternative"})
    pub = {}
    for pid, d in directory.items():
        card = d["public_card"]                     # agents see the other person's public profile only
        pub[pid] = {"persona_id": pid, "name": d["name"], "scenario": personas[pid]["scenario"], "public_text": card, "view": view}
    return {"personas": personas, "public_directory": pub, "requests": requests, "pairs": []}


def data_fingerprint(cdir: Path) -> dict[str, Any]:
    m = json.loads((cdir / "manifest.json").read_text(encoding="utf-8"))
    h = hashlib.sha256()
    for f in sorted((cdir / "owner_private").glob("u*.json")):
        h.update(f.read_bytes())
    h.update((cdir / "public_directory" / "directory.jsonl").read_bytes())
    return {"data_dir": str(cdir), "status": m.get("status"), "frozen": m.get("frozen"), "human_review": m.get("human_review"),
            "seed": m.get("seed"), "rules_version": m.get("rules_version"), "texts_sha256": h.hexdigest()[:16],
            "generation_status_counts": m.get("generation_status_counts")}


# ----------------------------------------------------------------------------- candidate pools

_TOK = re.compile(r"[a-z0-9]+")


def _tokens(t: str) -> list[str]:
    return _TOK.findall(t.lower())


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.ids = list(docs); self.toks = {i: _tokens(docs[i]) for i in self.ids}
        self.avg = sum(len(v) for v in self.toks.values()) / max(1, len(self.ids))
        self.df: Counter = Counter()
        for v in self.toks.values():
            self.df.update(set(v))
        self.n = len(self.ids); self.k1, self.b = k1, b

    def score(self, q: str, i: str) -> float:
        tf = Counter(self.toks[i]); dl = len(self.toks[i]); s = 0.0
        for w in set(_tokens(q)):
            if w not in tf:
                continue
            idf = math.log(1 + (self.n - self.df[w] + 0.5) / (self.df[w] + 0.5))
            s += idf * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * dl / self.avg))
        return s


async def ollama_embed(base_url: str, model: str, texts: list[str]) -> list[list[float]]:
    root = base_url.rsplit("/v1", 1)[0]
    async with httpx.AsyncClient(base_url=root, timeout=300, trust_env=False) as h:
        out = []
        for i in range(0, len(texts), 32):
            r = await h.post("/api/embed", json={"model": model, "input": texts[i:i + 32]})
            r.raise_for_status(); out += r.json()["embeddings"]
        return out


def _cos(a, b) -> float:
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


async def build_pools(inputs: dict[str, Any], k: int, embed_model: Optional[str], base_url: str,
                      request_ids: Optional[list[str]] = None) -> dict[str, Any]:
    """Pool(request) = union of BM25 top-k and embedding top-k over same-activity public cards. Fixed rule, no labels."""
    personas, pub = inputs["personas"], inputs["public_directory"]
    by_scen: dict[str, list[str]] = defaultdict(list)
    for pid, p in personas.items():
        by_scen[p["scenario"]].append(pid)
    bm = {sc: BM25({pid: pub[pid]["public_text"] for pid in ids}) for sc, ids in by_scen.items()}
    emb = None
    if embed_model:
        ids = list(pub); vecs = await ollama_embed(base_url, embed_model, [pub[i]["public_text"] for i in ids])
        emb = dict(zip(ids, vecs))
    pools = {}
    reqs = [r for r in inputs["requests"] if not request_ids or r["request_id"] in request_ids]
    qtexts = {}
    for r in reqs:
        p = personas[r["person_id"]]
        qtexts[r["request_id"]] = p["request_text"] if r["kind"] == "primary" else next(a["request_text"] for a in p["alt_requests"] if a["request_id"] == r["request_id"])
    qvecs = dict(zip(qtexts, await ollama_embed(base_url, embed_model, list(qtexts.values())))) if embed_model else {}
    for r in reqs:
        rid, pid = r["request_id"], r["person_id"]; sc = personas[pid]["scenario"]
        cands = [j for j in by_scen[sc] if j != pid]
        bm_scores = {j: round(bm[sc].score(qtexts[rid], j), 4) for j in cands}
        em_scores = {j: round(_cos(qvecs[rid], emb[j]), 4) for j in cands} if emb else {}
        bm_rank = sorted(cands, key=lambda j: -bm_scores[j])[:k]
        em_rank = sorted(cands, key=lambda j: -em_scores[j])[:k] if emb else []
        pool = list(dict.fromkeys(bm_rank + em_rank))
        pools[rid] = {"person_id": pid, "bm25_topk": bm_rank, "embed_topk": em_rank, "pool": pool,
                      "bm25_scores": {j: bm_scores[j] for j in pool}, "embed_scores": {j: em_scores.get(j) for j in pool}}
    return {"k": k, "embed_model": embed_model, "rule": "union(BM25 top-k, embedding top-k) over same-activity public cards", "pools": pools,
            "_vectors": {"cards": emb or {}, "requests": qvecs}}


# ----------------------------------------------------------------------------- direct LLM judge

class _DirectOut(BaseModel):
    decision: str = Field(description="recommend | reject | insufficient_info")
    reasons: str


class _CotCond(BaseModel):
    condition: str
    required: bool
    status: str = Field(description="satisfied | conflict | unknown")
    evidence: str


class _DirectCotOut(BaseModel):
    conditions: list[_CotCond]
    decision: str = Field(description="recommend | reject | insufficient_info")


_SYS_DIRECT_COT = """You are a matching assistant. First list every condition person A places on a partner (mark which are hard requirements), then check each one against what is visible about candidate B: satisfied (stated and inside what A accepts), conflict (stated and outside), unknown (not stated; never treat missing information as satisfied or as conflict). Then decide: reject if any hard requirement conflicts; insufficient_info if any hard requirement is unknown; otherwise recommend. Quote the words you rely on.
Return JSON: {"conditions": [...], "decision": "..."}"""


class _ScoreOut(BaseModel):
    score: int = Field(description="0-9: how well B fits what A is looking for right now")
    reasons: str


_SYS_SCORE = """You are a matching assistant. Rate from 0 to 9 how well candidate B fits what person A is looking for right now, using only what is visible about B. 0-3: something A requires is contradicted; 4-6: nothing contradicts but important things are unknown or preferences are unmet; 7-9: requirements met and preferences largely met. Return JSON: {"score": n, "reasons": "..."}"""


class _RankOut(BaseModel):
    ranking: list[str] = Field(description="candidate ids from best to worst fit")


_SYS_RANK = """You are a matching assistant. Given what person A is looking for right now and a list of candidates with what is visible about each, rank ALL candidate ids from best to worst fit. Return JSON: {"ranking": ["id", ...]}"""


async def llm_score_judge(client, a_request: str, b_card: str, rid: str, j: str):
    return await client.structured(_SYS_SCORE, f"A is looking for: {a_request}\n\nVisible about B: {b_card}", _ScoreOut, request_id=f"llm_score:{rid}:{j}")


async def llm_rank_pool(client, a_request: str, cards: dict[str, str], rid: str):
    listing = "\n\n".join(f"[{cid}] {txt}" for cid, txt in cards.items())
    return await client.structured(_SYS_RANK, f"A is looking for: {a_request}\n\nCandidates:\n{listing}", _RankOut, request_id=f"llm_rank:{rid}")


async def direct_cot_judge(client, a_request: str, b_card: str, rid: str, j: str):
    return await client.structured(_SYS_DIRECT_COT, f"A is looking for: {a_request}\n\nVisible about B: {b_card}", _DirectCotOut, request_id=f"direct_cot:{rid}:{j}")


_SYS_DIRECT = """You are a matching assistant. Given what person A is looking for RIGHT NOW and what is visible about candidate B, decide whether to introduce B to A.
- recommend: everything A requires is confirmed by what is visible about B;
- reject: something A requires is contradicted by what is visible about B;
- insufficient_info: something A requires is not stated in what is visible about B. Missing information is neither a match nor a mismatch.
Return JSON: {"decision": "...", "reasons": "..."}"""


async def direct_judge(client, a_request: str, b_card: str, rid: str, j: str):
    return await client.structured(_SYS_DIRECT, f"A is looking for: {a_request}\n\nVisible about B: {b_card}", _DirectOut, request_id=f"direct:{rid}:{j}")


# ----------------------------------------------------------------------------- run

def _registry_append(root: Path, rec: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "registry.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


async def run_community_experiment(cdir: Path, run_dir: Path, client, llm: LLMConfig, *, view: str, methods: list[str],
                                   pool_k: int, embed_model: Optional[str], request_kinds: list[str], subset: Optional[list[str]],
                                   purpose: str, resume: bool = True, max_calls: Optional[int] = None,
                                   shared_pools_dir: Optional[Path] = None) -> dict[str, Any]:
    inputs = load_community_inputs(cdir, view)
    vocab = load_vocab()
    run_dir.mkdir(parents=True, exist_ok=True)
    fp = data_fingerprint(cdir)
    meta = {"run_id": run_dir.name, "created_utc": datetime.now(timezone.utc).isoformat(), "purpose": purpose, "data": fp, "view": view,
            "methods": {m: METHODS.get(m, {"desc": m}) for m in methods}, "pool": {"k": pool_k, "embed_model": embed_model},
            "request_kinds": request_kinds, "subset": subset, "exp_version": EXP_VERSION, "agent_prompt_version": PROMPT_VERSION,
            "vocab_descriptions_in_prompts": VOCAB_DESCRIPTIONS, "model": llm.model, "params": {"temperature": llm.temperature, "seed": llm.seed,
            "max_tokens": llm.max_tokens, "reasoning_effort": llm.reasoning_effort}}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _registry_append(run_dir.parent, {k: meta[k] for k in ("run_id", "created_utc", "purpose", "data", "view", "methods", "pool", "request_kinds", "subset", "model")})
    # pools
    reqs = [r for r in inputs["requests"] if r["kind"] in request_kinds and (not subset or r["person_id"] in subset)]
    pool_path = run_dir / "pools.json"
    src_dir = shared_pools_dir or run_dir
    src_pool = src_dir / "pools.json"
    if not src_pool.exists():
        src_dir.mkdir(parents=True, exist_ok=True)
        all_reqs = [r for r in inputs["requests"] if r["kind"] in request_kinds] if shared_pools_dir else reqs
        pools = await build_pools(inputs, pool_k, embed_model, llm.base_url, [r["request_id"] for r in all_reqs])
        vec = pools.pop("_vectors", None)
        src_pool.write_text(json.dumps(pools, indent=1), encoding="utf-8")
        if vec:
            (src_dir / "embeddings.json").write_text(json.dumps(vec), encoding="utf-8")
    pools = json.loads(src_pool.read_text(encoding="utf-8"))
    if shared_pools_dir and not pool_path.exists():
        import shutil
        shutil.copy(src_pool, pool_path)
        if (src_dir / "embeddings.json").exists():
            shutil.copy(src_dir / "embeddings.json", run_dir / "embeddings.json")
    meta["shared_pools"] = str(src_dir)
    # E2: alternative requests reuse the PRIMARY request's pool (same candidates, only the request changes)
    prim_pool = {r["person_id"]: pools["pools"][r["request_id"]]["pool"] for r in inputs["requests"] if r["kind"] == "primary" and r["request_id"] in pools["pools"]}
    cases_path = run_dir / "cases.jsonl"
    done = set()
    if resume and cases_path.exists():
        for line in cases_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                if d.get("execution_status", "ok") == "ok":
                    done.add((d["request_id"], d["candidate_id"], d["method"]))
    cache_dir = run_dir.parent / "cache_community"
    key_parts = (fp["texts_sha256"], vocab.source_hash, PROMPT_VERSION, view, llm.model)
    n_ok = n_err = 0; t0 = time.perf_counter()
    with open(cases_path, "a", encoding="utf-8") as f:
        for r in reqs:
            rid, pid = r["request_id"], r["person_id"]
            pool = pools["pools"][rid]["pool"] if r["kind"] == "primary" else prim_pool.get(pid, pools["pools"].get(rid, {}).get("pool", []))
            p = inputs["personas"][pid]
            req_text = p["request_text"] if r["kind"] == "primary" else next(a["request_text"] for a in p["alt_requests"] if a["request_id"] == rid)
            # per-request agent inputs: the requester's CURRENT request is the one under test
            ai = json.loads(json.dumps(inputs))
            ai["personas"][pid]["request_text"] = req_text
            rank_cache: dict[str, Any] = {}
            if "llm_rank_top3" in methods and any((rid, j, "llm_rank_top3") not in done for j in pool) and pool:
                rr = await llm_rank_pool(client, req_text, {j: ai["public_directory"][j]["public_text"] for j in pool}, rid)
                rank_cache = {"ok": rr.ok, "error": rr.error, "ranking": [x for x in (rr.parsed.ranking if rr.ok else []) if x in pool],
                              "usage": {"calls": rr.calls, "prompt_tokens": rr.prompt_tokens, "completion_tokens": rr.completion_tokens, "latency_s": rr.wall_s}}
            for j in pool:
                for m in methods:
                    if (rid, j, m) in done:
                        continue
                    if max_calls is not None and client.budget.calls >= max_calls:
                        break
                    rec: dict[str, Any] = {"request_id": rid, "person_id": pid, "candidate_id": j, "method": m, "view": view,
                                           "request_kind": r["kind"], "finished_utc": None}
                    try:
                        if m in ("bm25_top3", "embed_top3"):
                            key_s = "bm25_scores" if m == "bm25_top3" else "embed_scores"
                            sc = pools["pools"].get(rid, {}).get(key_s) or {}
                            ranked = sorted(pool, key=lambda x: -(sc.get(x) or 0.0))
                            rec.update({"execution_status": "ok", "error": None, "decision_ab": "recommend" if j in ranked[:3] else "insufficient_info",
                                        "rank": ranked.index(j) + 1, "usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}})
                        elif m == "llm_score_top3":
                            res = await llm_score_judge(client, req_text, ai["public_directory"][j]["public_text"], rid, j)
                            rec.update({"execution_status": "ok" if res.ok else "execution_error", "error": res.error,
                                        "llm_score": res.parsed.score if res.ok else None, "decision_ab": None,   # decided offline: top-3 by score within pool
                                        "usage": {"calls": res.calls, "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens, "latency_s": res.wall_s}})
                        elif m == "llm_rank_top3":
                            rk = rank_cache.get("ranking", [])
                            rec.update({"execution_status": "ok" if rank_cache.get("ok") else "execution_error", "error": rank_cache.get("error"),
                                        "rank": (rk.index(j) + 1) if j in rk else None, "decision_ab": "recommend" if j in rk[:3] else "insufficient_info",
                                        "usage": rank_cache.get("usage") if j == pool[0] else {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}})
                        elif m == "direct_cot":
                            res = await direct_cot_judge(client, req_text, ai["public_directory"][j]["public_text"], rid, j)
                            rec.update({"execution_status": "ok" if res.ok else "execution_error", "error": res.error,
                                        "decision_ab": res.parsed.decision if res.ok else None, "conditions_cot": [c.model_dump() for c in res.parsed.conditions] if res.ok else None,
                                        "usage": {"calls": res.calls, "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens, "latency_s": res.wall_s}})
                        elif m == "direct":
                            res = await direct_judge(client, req_text, ai["public_directory"][j]["public_text"], rid, j)
                            rec.update({"execution_status": "ok" if res.ok else "execution_error", "error": res.error,
                                        "decision_ab": res.parsed.decision if res.ok else None, "reasons": res.parsed.reasons if res.ok else None,
                                        "usage": {"calls": res.calls, "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens, "latency_s": res.wall_s}})
                        else:
                            mr = await run_pair(ai, vocab, {"pair_id": f"{rid}|{j}", "requester_id": pid, "candidate_id": j}, client, method=m,
                                                max_questions=METHODS[m]["max_questions"], cache_dir=cache_dir, cache_key_parts=key_parts)
                            d = mr.model_dump(mode="json")
                            rec.update({"execution_status": d["execution_status"], "error": d["error"], "decision_ab": d["decision_ab"], "decision_ba": d["decision_ba"],
                                        "joint_decision": d["joint_decision"], "unresolved": d["unresolved"], "messages": d["messages"], "usage": d["usage"],
                                        "questions_used": (d.get("detail") or {}).get("questions_used"),
                                        "conditions_a": (d.get("detail") or {}).get("conditions_a"), "assess_a": (d.get("detail") or {}).get("assess_a"),
                                        "conditions_b": (d.get("detail") or {}).get("conditions_b"), "assess_b": (d.get("detail") or {}).get("assess_b"),
                                        "self_facts_a": (d.get("detail") or {}).get("self_facts_a"), "self_facts_b": (d.get("detail") or {}).get("self_facts_b")})
                    except Exception as e:
                        rec.update({"execution_status": "execution_error", "error": f"{type(e).__name__}:{e}"})
                    rec["finished_utc"] = datetime.now(timezone.utc).isoformat()
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                    if rec["execution_status"] == "ok": n_ok += 1
                    else: n_err += 1
    budget_errors = n_err if max_calls is not None and client.budget.calls >= max_calls else 0
    return {"run_dir": str(run_dir), "budget_cap_hit": budget_errors > 0, "requests": len(reqs), "pairs_planned": sum(len(pools["pools"][r["request_id"]]["pool"] if r["kind"] == "primary" else prim_pool.get(r["person_id"], [])) for r in reqs),
            "skipped_done": len(done), "ok": n_ok, "execution_error": n_err, "requests_sent": client.budget.calls, "wall_s": round(time.perf_counter() - t0, 1)}
