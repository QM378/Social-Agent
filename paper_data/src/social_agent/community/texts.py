"""Natural-language texts for the community: LLM writers produce each person's self-description and requests from
the structured notes; the other model screens them; assembly writes the owner and public layers that agents read."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from social_agent.community.generate import (ACTIVITY_ATTR, LENGTH_BANDS, CommunityPerson, RequestSpec, current_state,
                                          notes_for)
from social_agent.vocab import Vocab, load_vocab
from social_agent.llm import CallBudget, LLMConfig, make_client
from social_agent.schemas import Visibility

TEXT_PROMPT_VERSION = "community_text_v8"
CHECK_PROMPT_VERSION = "community_check_v3"
GENERATOR_RULES_VERSION = "community_rules_v2"   # bump when community.py generation/scoring rules change


def code_version() -> dict[str, str]:
    """Hashes of the rule-bearing source files, so a dataset can be tied to the exact generator code."""
    import social_agent.community.generate as cm
    out = {}
    for mod in (cm, __import__("social_agent.community.texts", fromlist=["x"]), __import__("social_agent.vocab", fromlist=["x"])):
        pth = Path(mod.__file__)
        out[pth.name] = hashlib.sha256(pth.read_bytes()).hexdigest()[:16]
    return out


async def model_provenance(llm: LLMConfig) -> dict[str, Any]:
    """Exact model identity from the serving endpoint (Ollama native API when present). Unknown fields stay 'unknown'."""
    import httpx
    prov: dict[str, Any] = {"served_model": llm.model, "base_url": llm.base_url, "digest": "unknown", "quantization": "unknown",
                            "family": "unknown", "parameter_size": "unknown", "modelfile_parameters": "unknown", "server": "unknown",
                            "base_model": "unknown"}
    root = llm.base_url.rsplit("/v1", 1)[0]
    try:
        async with httpx.AsyncClient(base_url=root, timeout=30, trust_env=False) as h:
            v = await h.get("/api/version")
            if v.status_code == 200:
                prov["server"] = f"ollama {v.json().get('version')}"
            tags = await h.get("/api/tags")
            if tags.status_code == 200:
                for m in tags.json().get("models", []):
                    if m.get("name") in (llm.model, f"{llm.model}:latest"):
                        prov["digest"] = m.get("digest", "unknown"); prov["modified_at"] = m.get("modified_at", "unknown")
                        d = m.get("details", {}); prov["quantization"] = d.get("quantization_level", "unknown")
                        prov["family"] = d.get("family", "unknown"); prov["parameter_size"] = d.get("parameter_size", "unknown")
            show = await h.post("/api/show", json={"model": llm.model})
            if show.status_code == 200:
                d = show.json(); prov["modelfile_parameters"] = d.get("parameters", "unknown")
                mf = d.get("modelfile", "")
                for line in mf.splitlines():
                    if line.upper().startswith("FROM "):
                        prov["base_model"] = line[5:].strip()
                prov["license_excerpt"] = (d.get("license") or "unknown")[:400]
    except Exception as e:
        prov["probe_error"] = f"{type(e).__name__}: {e}"
    return prov


class _Text(BaseModel):
    text: str


_STYLE_HINT = {"concise": "Short sentences, no fluff, like a quick profile.",
               "conversational": "Friendly and casual, first person, as if messaging a friend.",
               "narrative": "A bit of story: how you got into this, what a typical session looks like."}

_SYS_SELF = """You write a first-person self-description for a fictional person's partner-finding profile, in English.
Rules:
- EVERY fact in the notes must appear in the text, each with its meaning intact (days and times, pace, roles, teaching, communication, equipment opinions, venue, topic, game, mode). Do not drop, soften or strengthen any of them;
- do NOT ADD anything on those same topics that the notes do not state: no invented teaching/coaching/explaining, no talking or silence habits, no equipment opinions, no extra days, no skill claims. A "step-by-step walkthrough session" note only describes the session shape; never write that this person tutors, teaches, explains, coaches, guides, leads or is taught unless a separate note says so; do not use the words tutor or tutoring at all;
- allowed colour: the hobby given, why they enjoy the activity in general, mood, a place they like; one or two sentences at most;
- never mention who they want to meet or what they require from a partner (that goes elsewhere);
- write naturally; do not use technical labels, underscores, codes or list markers;
- target length {lo}-{hi} words (soft). Style: {style}
Return JSON: {{"text": "..."}}"""

_SYS_REQ = """You write, in English and first person, what a fictional person is looking for in a partner RIGHT NOW, from the notes.
Three strength levels; the reader must be able to tell them apart:
- MUST = a condition the partner has to meet or it is off: "I need ...", "you have to ...", "only if ...", "no ..., please", "it's a deal-breaker unless ...". Never "ideally", "would be great", "hopefully" for a MUST.
- STRONG WISH = matters a lot but not a deal-breaker: "I'd strongly prefer that ...", "it would matter a lot if ...", "big plus if ...", and it must END with a release such as "though it's not a requirement" or "but I won't rule you out over it". Never start a STRONG WISH or WISH with "I need" or "you must".
- WISH = light preference: "nice if ...", "a small bonus if ...", "ideally ..., but no big deal".
Other rules:
- vary the wording across items; do not use the same phrase for two items;
- the notes describe the PARTNER in neutral clauses ("you are available ...", "you are the one who coaches me"); each item lists its ALTERNATIVES: every alternative must be recognisably present in your sentence (all of them, joined by "or"); keep every negation; keep who does what exactly as written: "you are the one who coaches me" means the PARTNER coaches; "you play a support role" means the PARTNER plays support; a "walkthrough session" item says nothing about who explains, so do not assign that;
- if a THIS TIME note is present, say explicitly that it is a change from the usual and what the usual is;
- do not add requirements that are not in the notes; do not describe the person themselves beyond what the notes imply; do not mention what the person themselves can offer unless a note says so;
- 25-90 words, no labels, underscores, codes or list markers.
Return JSON: {{"text": "..."}}"""

_FORBIDDEN = re.compile(r"\b(time_window|comm_modes|team_role|coaching_role|editing_role|tutor_role|mutual_portraits|gear_talk|"
                        r"game_id|sat_morning|sat_afternoon|sun_morning|sun_afternoon|weekday_evening|casual_walk|occasional_talk|"
                        r"language_practice|in_person|Skyforge_Arena|Hollow_Lantern|MUST|WISH)\b")


def _atomic_write(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def expected_request_ids(priv: dict[str, Any], pid: str) -> list[str]:
    return [r["request_id"] for r in priv["requests"] if r["person_id"] == pid]


def effective_check_verdict(rec: Optional[dict[str, Any]], chk: Optional[dict[str, Any]], expected_rids: list[str]) -> str:
    """A check counts only if it was made on the CURRENT texts; a stale hash is reported as not_checked."""
    if not rec or not chk:
        return "not_checked"
    if chk.get("texts_hash") != texts_hash(rec, expected_rids):
        return "not_checked"
    return chk.get("verdict", "not_checked")


def text_record_complete(rec: Optional[dict[str, Any]], expected_rids: list[str]) -> bool:
    """A person's text record is complete only if the self text and EVERY expected request text are status ok."""
    if not rec or rec.get("self", {}).get("status") != "ok":
        return False
    return all(rec.get("requests", {}).get(rid, {}).get("status") == "ok" for rid in expected_rids)


def texts_hash(rec: dict[str, Any], expected_rids: list[str]) -> str:
    parts = [rec.get("self", {}).get("text", "")] + [rec.get("requests", {}).get(rid, {}).get("text", "") for rid in expected_rids]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def _wc(t: str) -> int:
    return len(t.split())


_MARKER_PATTERNS = {
    "teaching_role": r"\b(teach|teaching|coach|coaching|mentor|tutor|tutoring|explain|explaining|walk (you|me|them) through|take (you|me) through|learn from|guide|guiding|lead the|I follow|follow along|I offer|I can offer|I offer a|help (you|others|people)|share my (knowledge|skills|expertise)|show (you|others) how|learn from (you|others))\b",
    "communication": r"\b(voice chat|text chat|on mic|over voice|by text|discord|complete silence|in silence|silent while|silently|talk(ing)? (things )?through|short exchanges|no talking|chatting while)\b",
    "equipment": r"\b(gear|equipment|lens|lenses|camera talk|settings|rig)\b",
    "schedule": r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekday|weekend|morning|afternoon|evening|night)\b",
    "pace": r"\b(slow|slowly|fast|quick|quickly|pace|rush|relaxed pace|leisurely)\b",
    "role": r"\b(support|damage|dps|tank|healer|flexible|any (role|position))\b",
}


def review_markers(text: str, allowed_topics: set[str]) -> list[str]:
    """Keyword markers for match-relevant wording in a self text whose visible facts do not cover that topic.
    Markers are for human review only; they never reject a text."""
    return [t for t, _ in review_marker_hits(text, allowed_topics)]


def review_marker_hits(text: str, allowed_topics: set[str]) -> list[tuple[str, str]]:
    """(topic, offending sentence) pairs, so a repair prompt can point at the exact sentence."""
    import re as _re
    hits = []
    for topic, pat in _MARKER_PATTERNS.items():
        if topic in allowed_topics:
            continue
        m = _re.search(pat, text, _re.I)
        if m:
            start = text.rfind(".", 0, m.start()) + 1; end = text.find(".", m.end())
            hits.append((topic, text[start: end + 1 if end != -1 else None].strip()))
    return hits


_TOPIC_OF_ATTR = {"editing_role": "teaching_role", "coaching_role": "teaching_role", "tutor_role": "teaching_role",
                  "comm_modes": "communication", "quiet": "communication", "gear_talk": "equipment", "time_window": "schedule",
                  "pace": "pace", "team_role": "role"}


_STOP = {"the", "and", "with", "that", "this", "from", "your", "you", "for", "not", "but", "are", "our", "one", "each", "other", "while",
         "working", "session", "things", "some", "any", "all", "role", "into", "about", "what", "when", "there", "them", "have", "just", "only"}


def fact_coverage_issues(text: str, fact_sentences: list[str]) -> list[str]:
    """Keyword heuristic: each visible fact sentence should leave at least one distinctive content word in the text.
    Misses come back as 'fact_not_mentioned:<sentence>' so the repair prompt can list them. Never a rejection by itself."""
    import re as _re
    low = text.lower()
    miss = []
    for sent in fact_sentences:
        words = [w for w in _re.findall(r"[a-z]+", sent.lower()) if len(w) > 3 and w not in _STOP]
        if words and not any(w in low or (w.endswith("s") and w[:-1] in low) or (w[:-1] in low and len(w) > 5) for w in words):
            miss.append(sent)
    return [f"fact_not_mentioned:{m}" for m in miss]


def _structural_issues(kind: str, text: str, band: tuple[int, int] | None) -> list[str]:
    issues = []
    if not text or _wc(text) < 15:
        issues.append("too_short")
    if _FORBIDDEN.search(text):
        issues.append("label_or_code_leaked")
    if kind == "self" and band:
        lo, hi = band
        if _wc(text) < lo * 0.5 or _wc(text) > hi * 1.6:
            issues.append(f"length_out_of_band:{_wc(text)}")
    if kind == "request" and (_wc(text) < 20 or _wc(text) > 140):
        issues.append(f"request_length:{_wc(text)}")
    if re.search(r"^\s*[-*\d]+[.)]?\s", text, re.M):
        issues.append("list_markers")
    return issues


# ----------------------------------------------------------------------------- assignment

def assign_models(comm: dict[str, Any], model_keys: list[str], seed: int) -> dict[str, str]:
    """Stratified (scenario x length band x required-count bucket) round-robin assignment; both requests of a person share a model."""
    import random
    rng = random.Random(seed)
    people = comm["people"]; reqs = comm["requests"]
    nreq = {r["person_id"]: sum(q["required"] for q in r["requirements"]) for r in reqs if r["kind"] == "primary"}
    strata: dict[tuple, list[str]] = defaultdict(list)
    for p in people:
        strata[(p["scenario"], p["length_band"], "1" if nreq[p["person_id"]] <= 1 else "2-3" if nreq[p["person_id"]] <= 3 else "4")].append(p["person_id"])
    assignment = {}
    for key in sorted(strata):
        ids = strata[key]; rng.shuffle(ids)
        offset = rng.randrange(len(model_keys))
        for i, pid in enumerate(ids):
            assignment[pid] = model_keys[(i + offset) % len(model_keys)]
    return assignment


# ----------------------------------------------------------------------------- generation

async def _gen_item(client, system: str, user: str, kind: str, band, item_seed: int, max_repairs: int = 2,
                    fact_sentences: list[str] | None = None, withheld_topics: set[str] | None = None) -> dict[str, Any]:
    attempts = []
    sys_now = system
    for k in range(1 + max_repairs):
        client.cfg.seed = item_seed + k
        res = await client.structured(sys_now, user, _Text, request_id=f"gen:{kind}:{item_seed}:{k}")
        rec = res.to_record()
        text = res.parsed.text.strip() if res.ok else ""
        issues = _structural_issues(kind, text, band) if res.ok else [f"llm_error:{res.error}"]
        if res.ok and fact_sentences:
            issues += fact_coverage_issues(text, fact_sentences)
        if res.ok and withheld_topics:
            for topic, sent in review_marker_hits(text, set()):
                if topic in withheld_topics:
                    issues.append(f"withheld_topic_wording:{topic}: the notes say nothing about {topic.replace('_', ' ')}; "
                                  f"DELETE or neutralise this sentence so it no longer says or implies it: \"{sent}\"")
        attempts.append({"attempt": k, "seed": item_seed + k, "text": text, "issues": issues, "ok": res.ok, "calls": rec["calls"],
                         "prompt_tokens": rec["prompt_tokens"], "completion_tokens": rec["completion_tokens"],
                         "truncated": rec["truncated"], "wall_s": rec["wall_s"], "error": rec["error"],
                         "prompt_sent": {"system": sys_now, "user": user}, "raw_output": rec["raw_text"],
                         "reasoning_returned": bool(rec.get("reasoning_text")),
                         "http_attempts": rec["attempts"]})      # every real request: exact messages, params, response
        if res.ok and not issues:
            return {"status": "ok", "text": text, "attempts": attempts}
        fix = [i for i in issues if not i.startswith("llm_error")]
        sys_now = system + f"\n\nYour previous draft had these problems: {fix}. Keep everything else, fix only these, and return only JSON."
    last_ok = [a for a in attempts if a["ok"]]
    return {"status": "needs_review" if last_ok else "failed", "text": last_ok[-1]["text"] if last_ok else "", "attempts": attempts}


def _person_user_block(p: CommunityPerson, notes: dict[str, Any]) -> str:
    bg = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in p.background.items())
    return (f"Name: {p.name}\nAge: {p.age_band}. Gender: {p.gender}. Nationality: {p.nationality}. Hobby outside this: {p.hobby}.\n"
            f"Background: {bg}\nFacts (stable):\n" + "\n".join(f"- {t}" for t in notes["self_notes"]))


def _request_user_block(p: CommunityPerson, notes: dict[str, Any]) -> str:
    lines = [f"- {n['strength']}: {n['text']}" + (f"   [alternatives: {' | '.join(n['alternatives'])}]" if len(n.get("alternatives", [])) > 1 else "")
             for n in notes["request_notes"]]
    tt = ("\nTHIS TIME: " + "; ".join(notes["this_time"])) if notes["this_time"] else ""
    return f"Activity: {p.scenario} partner\nRequirements:\n" + "\n".join(lines) + tt


async def render_community(out: Path, model_key: str, llm: LLMConfig, only_ids: Optional[list[str]] = None,
                           limit: Optional[int] = None, max_calls: Optional[int] = None) -> dict[str, Any]:
    priv = json.loads((out / "evaluator" / "community_private.json").read_text(encoding="utf-8"))
    notes_all = json.loads((out / "generation" / "notes.json").read_text(encoding="utf-8"))
    assignment = json.loads((out / "generation" / "assignment.json").read_text(encoding="utf-8"))
    people = {p["person_id"]: CommunityPerson.model_validate(p) for p in priv["people"]}
    reqs_by = defaultdict(list)
    for r in priv["requests"]:
        reqs_by[r["person_id"]].append(RequestSpec.model_validate(r))
    todo = [pid for pid, mk in assignment.items() if mk == model_key and (not only_ids or pid in only_ids)]
    tdir = out / "generation" / "texts"; tdir.mkdir(parents=True, exist_ok=True)
    prov = await model_provenance(llm)
    prov.update({"model_key": model_key, "params": {"temperature": llm.temperature, "max_tokens": llm.max_tokens, "reasoning_effort": llm.reasoning_effort,
                                                      "json_schema_mode": llm.json_schema_mode, "max_repairs": llm.max_repairs},
                 "prompt_version": TEXT_PROMPT_VERSION, "rules_version": GENERATOR_RULES_VERSION, "code_version": code_version(),
                 "recorded_utc": datetime.now(timezone.utc).isoformat()})
    _atomic_write(out / "generation" / f"provenance_{model_key}.json", prov)
    client = make_client(llm, budget=CallBudget(max_calls))
    done = failed = review = skipped = 0
    t0 = time.perf_counter()
    try:
        for pid in todo[:limit] if limit else todo:
            f = tdir / f"{pid}.json"
            exp = [r.request_id for r in reqs_by[pid]]
            prev = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
            if text_record_complete(prev, exp):
                skipped += 1; continue
            p = people[pid]
            band = LENGTH_BANDS[p.length_band]
            base_seed = int(hashlib.sha256(f"{pid}:{TEXT_PROMPT_VERSION}".encode()).hexdigest()[:8], 16) % 10_000_000
            notes_primary = notes_all[f"{pid}.r1"]
            withheld_topics = {_TOPIC_OF_ATTR[a] for a, f in p.facts.items() if a in _TOPIC_OF_ATTR and f.visibility == Visibility.withheld} \
                - {_TOPIC_OF_ATTR[a] for a, f in p.facts.items() if a in _TOPIC_OF_ATTR and f.visibility != Visibility.withheld}
            self_res = await _gen_item(client, _SYS_SELF.format(lo=band[0], hi=band[1], style=_STYLE_HINT[p.text_style]),
                                       _person_user_block(p, notes_primary), "self", band, base_seed, max_repairs=3,
                                       fact_sentences=notes_primary["self_notes"], withheld_topics=withheld_topics)
            self_res["residual_withheld_wording"] = [i for i in self_res["attempts"][-1]["issues"] if i.startswith("withheld_topic_wording")] if self_res["attempts"] else []
            allowed = {_TOPIC_OF_ATTR[a] for a, f in p.facts.items() if a in _TOPIC_OF_ATTR and f.visibility != Visibility.withheld}
            self_res["review_markers"] = review_markers(self_res.get("text", ""), allowed)
            req_res = {}
            for r in reqs_by[pid]:
                alts = [a for n in notes_all[r.request_id]["request_notes"] for a in n.get("alternatives", []) if len(n.get("alternatives", [])) > 1]
                req_res[r.request_id] = await _gen_item(client, _SYS_REQ, _request_user_block(p, notes_all[r.request_id]), "request", None,
                                                        base_seed + (17 if r.kind == "alternative" else 3), fact_sentences=alts)
            # reuse pieces that were already ok in a previous partial record; keep the previous record in history
            if prev:
                if prev.get("self", {}).get("status") == "ok" and self_res["status"] != "ok":
                    self_res = prev["self"]
                for rid in exp:
                    if prev.get("requests", {}).get(rid, {}).get("status") == "ok" and req_res[rid]["status"] != "ok":
                        req_res[rid] = prev["requests"][rid]
            history = (prev.get("history", []) if prev else []) + ([{k: v for k, v in prev.items() if k != "history"}] if prev else [])
            rec = {"person_id": pid, "model_key": model_key, "model": llm.model, "model_digest": prov.get("digest", "unknown"),
                   "model_quantization": prov.get("quantization", "unknown"), "prompt_version": TEXT_PROMPT_VERSION,
                   "rules_version": GENERATOR_RULES_VERSION, "code_version": code_version(),
                   "params": {"temperature": llm.temperature, "max_tokens": llm.max_tokens, "reasoning_effort": llm.reasoning_effort},
                   "generated_utc": datetime.now(timezone.utc).isoformat(), "self": self_res, "requests": req_res,
                   "complete": text_record_complete({"self": self_res, "requests": req_res}, exp),
                   "generation_status": "complete" if text_record_complete({"self": self_res, "requests": req_res}, exp)
                   else "failed" if self_res["status"] == "failed" or any(x["status"] == "failed" for x in req_res.values()) else "needs_review",
                   "auto_check_status": "not_checked", "human_review_status": "not_reviewed", "history": history}
            _atomic_write(f, rec)
            st = [self_res["status"]] + [x["status"] for x in req_res.values()]
            if all(s == "ok" for s in st): done += 1
            elif "failed" in st: failed += 1
            else: review += 1
    finally:
        await client.aclose()
    return {"model_key": model_key, "model": llm.model, "planned": len(todo), "done": done, "needs_review": review, "failed": failed,
            "skipped_done": skipped, "usable": done + review + skipped, "requests_sent": client.budget.calls,
            "wall_s": round(time.perf_counter() - t0, 1)}


# ----------------------------------------------------------------------------- cross-check (other model)

class _ExtractedFacts(BaseModel):
    facts: dict[str, str | list[str]]


class _ExtReq(BaseModel):
    attribute: str
    accepted_values: list[str]
    strength: str = Field(description="required | strong_preference | preference")


class _ExtThisTime(BaseModel):
    attribute: str
    value: str = Field(description="vocabulary value the person says applies THIS TIME (a change from usual)")


class _ExtReqs(BaseModel):
    requirements: list[_ExtReq]
    changes_this_time: list[_ExtThisTime] = Field(default_factory=list)


def _nv(v):
    """Normalize checker output: single-element list -> value; [] / ['unknown'] -> 'unknown'."""
    if isinstance(v, list):
        vals = [x for x in v if x != "unknown"]
        if not vals:
            return "unknown"
        return vals[0] if len(vals) == 1 else sorted(vals)
    return v


# Impact classes (what a difference would change if the text were taken at face value):
#   decision    -> could flip accept / reject for some pair
#   ranking     -> could change scores or order among acceptable candidates
#   information -> could change what a counterpart can learn (missing fact, leaked withheld fact, extra 'this time' state)
#   wording     -> tone / phrasing only
# Every class except wording goes to the human adjudication list. The class is about impact, NOT about whether
# the checker is right: a decision-class item can still be a checker misreading; adjudication settles that.
IMPACT_RANK = {"decision": 3, "information": 2, "ranking": 1, "wording": 0}


def compare_facts(p: CommunityPerson, got: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for a, fct in p.facts.items():
        g = _nv(got.get(a, "unknown"))
        if fct.visibility == Visibility.withheld:
            if g != "unknown":
                out.append({"where": "self", "type": "withheld_fact_present", "attribute": a, "got": g, "severity": "information"})
            continue
        exp = _nv(fct.value)
        if g == "unknown":
            out.append({"where": "self", "type": "fact_missing", "attribute": a, "expected": exp, "severity": "information"}); continue
        gs = set(g) if isinstance(g, list) else {g}; es = set(exp) if isinstance(exp, list) else {exp}
        if gs == es:
            continue
        if isinstance(fct.value, list) and gs < es:
            out.append({"where": "self", "type": "fact_subset", "attribute": a, "expected": exp, "got": g, "severity": "information"})
        else:
            out.append({"where": "self", "type": "fact_changed", "attribute": a, "expected": exp, "got": g, "severity": "decision"})
    return out


def compare_requirements(gold: RequestSpec, got: Any) -> list[dict[str, Any]]:
    rid = gold.request_id; out = []
    gotd = {q.attribute: q for q in got.requirements}
    for q in gold.requirements:
        g = gotd.get(q.attribute)
        if g is None:
            out.append({"where": rid, "type": "requirement_missing", "attribute": q.attribute, "required": q.required,
                        "severity": "decision" if q.required else "ranking"}); continue
        ga, ea = set(g.accepted_values), set(q.accepted_values)
        if ga != ea:          # a narrowed or widened accepted set changes outcomes for a required condition, ranking for a preference
            out.append({"where": rid, "type": "accepted_set_differs", "attribute": q.attribute, "expected": sorted(ea), "got": sorted(ga),
                        "severity": "decision" if q.required else "ranking"})
        gs = g.strength
        if q.required and gs != "required":
            out.append({"where": rid, "type": "strength_changed", "attribute": q.attribute, "expected": "required", "got": gs, "severity": "decision"})
        elif not q.required and gs == "required":
            out.append({"where": rid, "type": "strength_changed", "attribute": q.attribute, "expected": "preference", "got": gs, "severity": "decision"})
        elif not q.required:
            strong = gold.weights.get(q.id, 1.0) >= 2.0
            if strong != (gs == "strong_preference"):
                out.append({"where": rid, "type": "weight_changed", "attribute": q.attribute, "expected": "strong_preference" if strong else "preference",
                            "got": gs, "severity": "ranking"})
    for a in set(gotd) - {q.attribute for q in gold.requirements}:
        vocab_attr = a in {q.attribute for q in gold.requirements} or True
        out.append({"where": rid, "type": "requirement_added", "attribute": a, "got": gotd[a].accepted_values,
                    "severity": "decision" if gotd[a].strength == "required" else "ranking"})
    gt = {c.attribute: c.value for c in got.changes_this_time}
    for a, v in gold.overrides.items():
        if a not in gt:
            out.append({"where": rid, "type": "this_time_missing", "attribute": a, "expected": v, "severity": "decision"})
        elif gt[a] != v:
            out.append({"where": rid, "type": "this_time_wrong", "attribute": a, "expected": v, "got": gt[a], "severity": "decision"})
    for a in set(gt) - set(gold.overrides):
        out.append({"where": rid, "type": "this_time_added", "attribute": a, "got": gt[a], "severity": "information"})
    return out


def verdict_from(issues: list[dict[str, Any]], checked: dict[str, bool], checker_errors: int) -> str:
    """clean | needs_adjudication (any decision/information/ranking item) | wording_only | incomplete | checker_error."""
    if checker_errors:
        return "checker_error"
    if not all(checked.values()):
        return "incomplete"
    sem = [i for i in issues if i["type"] != "checker_error"]
    if not sem:
        return "clean"
    return "needs_adjudication" if any(IMPACT_RANK.get(i.get("severity", "wording"), 0) > 0 for i in sem) else "wording_only"


def _vocab_glossary(vocab: Vocab, scenario: str) -> str:
    s = vocab.scenarios[scenario]
    return "\n".join(f"- {a} ({'set' if sp.kind == 'set' else 'single'}): " + "; ".join(f"{v} = {sp.meaning.get(v, v)}" for v in sp.values)
                     for a, sp in s.attributes.items())


async def crosscheck_community(out: Path, checker_key: str, llm: LLMConfig, max_calls: Optional[int] = None,
                               only_model_key: Optional[str] = None) -> dict[str, Any]:
    """The checker model extracts facts/requirements from the generated texts of the OTHER generator and compares to gold.
    Output is an issue list per person; disputes are flagged, never auto-rewritten."""
    priv = json.loads((out / "evaluator" / "community_private.json").read_text(encoding="utf-8"))
    vocab = load_vocab()
    assignment = json.loads((out / "generation" / "assignment.json").read_text(encoding="utf-8"))
    people = {p["person_id"]: CommunityPerson.model_validate(p) for p in priv["people"]}
    reqs = {r["request_id"]: RequestSpec.model_validate(r) for r in priv["requests"]}
    tdir = out / "generation" / "texts"; cdir = out / "generation" / "crosscheck"; cdir.mkdir(parents=True, exist_ok=True)
    client = make_client(llm, budget=CallBudget(max_calls))
    cprov = await model_provenance(llm); cprov.update({"model_key": checker_key, "prompt_version": CHECK_PROMPT_VERSION, "temperature": llm.temperature})
    _atomic_write(cdir / f"_checker_provenance_{checker_key}.json", cprov)
    n = 0; issues_total = Counter(); verdicts = Counter()
    try:
        for f in sorted(x for x in tdir.glob("*.json") if not x.name.startswith("_")):
            rec = json.loads(f.read_text(encoding="utf-8")); pid = rec["person_id"]
            if rec["model_key"] == checker_key or (only_model_key and rec["model_key"] != only_model_key):
                continue
            cf = cdir / f"{pid}.json"
            exp = expected_request_ids(priv, pid)
            th = texts_hash(rec, exp)
            if cf.exists():
                old = json.loads(cf.read_text(encoding="utf-8"))
                if old.get("texts_hash") == th and old.get("verdict") in ("clean", "dispute", "hard_dispute", "soft_only", "needs_adjudication", "wording_only"):
                    continue                                   # same texts, already fully checked
            p = people[pid]; gl = _vocab_glossary(vocab, p.scenario)
            checker_attempts = []
            out_rec = {"person_id": pid, "generator": rec["model_key"], "checker": checker_key, "checker_model": llm.model,
                       "prompt_version": CHECK_PROMPT_VERSION, "issues": [], "texts_hash": th,
                       "checked": {"self": False, **{rid: False for rid in exp}}, "checker_errors": 0}
            if rec["self"]["status"] != "failed" and rec["self"]["text"]:
                r1 = await client.structured("Extract this person's own facts using only the vocabulary. \"unknown\" if not stated. Set attributes return a list. "
                                             "Return {\"facts\": {...}}.\n" + gl, rec["self"]["text"], _ExtractedFacts, request_id=f"check_facts:{pid}")
                checker_attempts.append({"item": "self", "attempts": r1.to_record()["attempts"]})
                if r1.ok:
                    out_rec["checked"]["self"] = True
                    out_rec["issues"] += compare_facts(p, r1.parsed.facts)
                else:
                    out_rec["checker_errors"] += 1
                    out_rec["issues"].append({"where": "self", "type": "checker_error", "error": r1.error})
            for rid in exp:
                rr = rec["requests"].get(rid, {"status": "failed", "text": ""})
                if rr["status"] == "failed" or not rr["text"]:
                    continue
                r2 = await client.structured("Extract (1) the requirements this person places on a partner: attribute, accepted_values (vocabulary values the PARTNER must hold), "
                                             "strength = required ONLY when the wording is unconditional (must / need / have to / only / no ... please / deal-breaker unless); "
                                             "strong_preference when it is called important but explicitly not required ('really matters ... though not a requirement', 'strongly prefer', 'big plus'); "
                                             "preference for light wishes ('nice if', 'small bonus', 'ideally'). Who does what: 'you to be the one who coaches me' means the PARTNER coaches (partner value teach); "
                                             "(2) changes_this_time: anything the person says is different from usual for THIS occasion, as attribute + their own value this time "
                                             "(empty list if none). Return {\"requirements\": [...], \"changes_this_time\": [...]}.\n" + gl,
                                             rr["text"], _ExtReqs, request_id=f"check_req:{rid}")
                checker_attempts.append({"item": rid, "attempts": r2.to_record()["attempts"]})
                if not r2.ok:
                    out_rec["checker_errors"] += 1
                    out_rec["issues"].append({"where": rid, "type": "checker_error", "error": r2.error}); continue
                out_rec["checked"][rid] = True
                out_rec["issues"] += compare_requirements(reqs[rid], r2.parsed)
            out_rec["n_issues"] = sum(1 for i in out_rec["issues"] if i["type"] != "checker_error")
            out_rec["impact_counts"] = dict(Counter(i.get("severity", "wording") for i in out_rec["issues"] if i["type"] != "checker_error"))
            out_rec["adjudication"] = {"status": "pending", "by": None, "notes": ""}
            out_rec["verdict"] = verdict_from(out_rec["issues"], out_rec["checked"], out_rec["checker_errors"])
            out_rec["checker_provenance"] = cprov
            out_rec["checker_http_attempts"] = checker_attempts
            _atomic_write(cf, out_rec)
            rec["auto_check_status"] = out_rec["verdict"]; rec["auto_check_texts_hash"] = th
            _atomic_write(f, rec)
            n += 1; verdicts[out_rec["verdict"]] += 1
            for i in out_rec["issues"]:
                issues_total[i["type"]] += 1
    finally:
        await client.aclose()
    return {"checker_key": checker_key, "checked": n, "verdicts": dict(verdicts), "issue_counts": dict(issues_total), "requests_sent": client.budget.calls,
            "note": "automatic screen; a dispute means generator text and checker reading disagree, human review decides"}


def recheck_offline(out: Path) -> dict[str, Any]:
    """Recompute issues and verdicts from the STORED checker responses (no LLM call): used when the comparison rules change."""
    priv = json.loads((out / "evaluator" / "community_private.json").read_text(encoding="utf-8"))
    people = {p["person_id"]: CommunityPerson.model_validate(p) for p in priv["people"]}
    reqs = {r["request_id"]: RequestSpec.model_validate(r) for r in priv["requests"]}
    cdir = out / "generation" / "crosscheck"; verdicts = Counter(); n = 0
    for cf in sorted(x for x in cdir.glob("u*.json")):
        c = json.loads(cf.read_text(encoding="utf-8")); pid = c["person_id"]
        issues = []; checked = {k: False for k in c.get("checked", {})}; errors = 0
        for att in c.get("checker_http_attempts", []):
            oks = [a for a in att["attempts"] if a["status"] == "ok" and a.get("response_content")]
            if not oks:
                errors += 1; continue
            try:
                got = json.loads(oks[-1]["response_content"])
                if att["item"] == "self":
                    issues += compare_facts(people[pid], _ExtractedFacts.model_validate(got).facts)
                else:
                    issues += compare_requirements(reqs[att["item"]], _ExtReqs.model_validate(got))
                checked[att["item"]] = True
            except Exception as e:
                errors += 1; issues.append({"where": att["item"], "type": "checker_error", "error": f"parse:{e}"})
        c["issues"] = issues; c["checked"] = checked; c["checker_errors"] = errors
        c["n_issues"] = sum(1 for i in issues if i["type"] != "checker_error")
        c["impact_counts"] = dict(Counter(i.get("severity", "wording") for i in issues if i["type"] != "checker_error"))
        c["top_impact"] = max((i.get("severity", "wording") for i in issues if i["type"] != "checker_error"), key=lambda x: IMPACT_RANK.get(x, 0), default="none")
        c["adjudication"] = c.get("adjudication", {"status": "pending", "by": None, "notes": ""})
        c["verdict"] = verdict_from(issues, checked, errors); c["recheck_rules"] = CHECK_PROMPT_VERSION + "+offline_v2"
        _atomic_write(cf, c); verdicts[c["verdict"]] += 1; n += 1
        tf = out / "generation" / "texts" / f"{pid}.json"
        if tf.exists():
            rec = json.loads(tf.read_text(encoding="utf-8")); rec["auto_check_status"] = c["verdict"]; _atomic_write(tf, rec)
    return {"rechecked": n, "verdicts": dict(verdicts)}


# ----------------------------------------------------------------------------- assembly (owner and public layers)

def assemble_community(out: Path) -> dict[str, Any]:
    priv = json.loads((out / "evaluator" / "community_private.json").read_text(encoding="utf-8"))
    vocab = load_vocab()
    notes_all = json.loads((out / "generation" / "notes.json").read_text(encoding="utf-8"))
    people = {p["person_id"]: CommunityPerson.model_validate(p) for p in priv["people"]}
    reqs = [RequestSpec.model_validate(r) for r in priv["requests"]]
    tdir = out / "generation" / "texts"; cdir = out / "generation" / "crosscheck"
    status = Counter(); pub_rows = []
    for pid, p in people.items():
        f = tdir / f"{pid}.json"
        rec = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        chk = json.loads((cdir / f"{pid}.json").read_text(encoding="utf-8")) if (cdir / f"{pid}.json").exists() else None
        my_reqs = [r for r in reqs if r.person_id == pid]
        n1 = notes_all[f"{pid}.r1"]
        self_text = rec["self"]["text"] if rec and rec["self"]["text"] else None
        req_texts = {r.request_id: (rec["requests"][r.request_id]["text"] if rec and rec["requests"].get(r.request_id, {}).get("text") else None) for r in my_reqs}
        exp = [r.request_id for r in my_reqs]
        items = ([rec["self"]] + [rec["requests"].get(rid, {"status": "failed"}) for rid in exp]) if rec else []
        text_status = "missing" if not rec else ("ok" if text_record_complete(rec, exp) else "failed" if any(x["status"] == "failed" for x in items) else "needs_review")
        status[text_status] += 1
        pub_rows.append({"person_id": pid, "name": p.name, "interaction_type": f"{p.scenario}_partner", "public_card": n1["template_public"],
                         "public_card_source": "template_public_facts"})
        (out / "owner_private").mkdir(exist_ok=True)
        (out / "owner_private" / f"{pid}.json").write_text(json.dumps({
            "person_id": pid, "name": p.name, "interaction_type": f"{p.scenario}_partner",
            "self_text": self_text, "self_text_source": rec["model_key"] if rec else None,
            "requests": [{"request_id": r.request_id, "kind": r.kind, "request_text": req_texts[r.request_id],
                          "this_time": notes_all[r.request_id]["this_time"]} for r in my_reqs],
            "disclosure_policy": {"public_evidence_ids": [f.evidence_id for f in p.facts.values() if f.visibility == Visibility.public],
                                  "disclosable_evidence_ids": [f.evidence_id for f in p.facts.values() if f.visibility == Visibility.disclosable]},
            "generation_status": text_status, "auto_check_status": effective_check_verdict(rec, chk, exp),
            "human_review_status": (rec or {}).get("human_review_status", "not_reviewed"),
            "provenance": {"generator": rec["model_key"] if rec else None, "model": rec["model"] if rec else None,
                           "model_digest": rec.get("model_digest", "unknown") if rec else None, "prompt_version": rec["prompt_version"] if rec else None,
                           "rules_version": rec.get("rules_version") if rec else None}}, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(out / "public_directory" / "directory.jsonl", "w", encoding="utf-8") as f:
        for row in pub_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({"text_mode": "llm_mixed" if status.get("ok") else "none_yet", "generation_status_counts": dict(status),
                     "assembled_utc": datetime.now(timezone.utc).isoformat(), "pilot": True, "status": "review_pending", "frozen": False,
                     "human_review": "not_reviewed", "rules_version": GENERATOR_RULES_VERSION, "code_version": code_version(),
                     "label_note": "scores are compatibility under the published synthetic rules, not real human preference",
                     "statuses_are_independent": "generation ok != auto check clean != human review passed"})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def summary(out: Path) -> dict[str, Any]:
    """Final status: complete / needs_review / failed / missing people, cross-check verdicts, cost. `all_complete` is the gate."""
    priv = json.loads((out / "evaluator" / "community_private.json").read_text(encoding="utf-8"))
    tdir = out / "generation" / "texts"; cdir = out / "generation" / "crosscheck"
    st = Counter(); cost = Counter(); by_model = Counter(); verdicts = Counter(); wall = 0.0
    for p in priv["people"]:
        pid = p["person_id"]; f = tdir / f"{pid}.json"; exp = expected_request_ids(priv, pid)
        if not f.exists():
            st["missing"] += 1; continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        items = [rec["self"]] + [rec["requests"].get(rid, {"status": "failed", "attempts": []}) for rid in exp]
        s = "complete" if text_record_complete(rec, exp) else "failed" if any(x["status"] == "failed" for x in items) else "needs_review"
        st[s] += 1; by_model[rec["model_key"]] += 1
        for x in items:
            for a in x.get("attempts", []):
                cost["calls"] += a["calls"]; cost["prompt_tokens"] += a["prompt_tokens"] or 0; cost["completion_tokens"] += a["completion_tokens"] or 0; wall += a["wall_s"]
        cf = cdir / f"{pid}.json"
        chk = json.loads(cf.read_text(encoding="utf-8")) if cf.exists() else None
        verdicts[effective_check_verdict(rec, chk, exp)] += 1
    n = len(priv["people"])
    checked_ok = sum(v for k, v in verdicts.items() if k in ("clean", "dispute", "hard_dispute", "soft_only", "needs_adjudication", "wording_only"))
    return {"n_people": n, "people_text_status": dict(st), "by_generator": dict(by_model), "generation_cost": dict(cost),
            "generation_wall_s": round(wall, 1), "crosscheck_verdicts": dict(verdicts),
            "all_complete": st.get("complete", 0) == n, "all_checked_clean_or_dispute": checked_ok == n}


def model_status(out: Path, model_key: str) -> dict[str, Any]:
    """Machine-readable gate for the batch script: how many people assigned to model_key have usable (complete or reviewable) texts."""
    priv = json.loads((out / "evaluator" / "community_private.json").read_text(encoding="utf-8"))
    assignment = json.loads((out / "generation" / "assignment.json").read_text(encoding="utf-8"))
    tdir = out / "generation" / "texts"
    c = Counter()
    for pid, mk in assignment.items():
        if mk != model_key:
            continue
        f = tdir / f"{pid}.json"; exp = expected_request_ids(priv, pid)
        if not f.exists():
            c["missing"] += 1; continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        items = [rec["self"]] + [rec["requests"].get(rid, {"status": "failed"}) for rid in exp]
        c["complete" if text_record_complete(rec, exp) else "failed" if any(x["status"] == "failed" for x in items) else "needs_review"] += 1
        if rec["self"].get("residual_withheld_wording"):
            c["residual_withheld_wording"] += 1
    total = c["complete"] + c["needs_review"] + c["failed"] + c["missing"]
    return {"model_key": model_key, "assigned": total, **dict(c), "usable": c["complete"] + c["needs_review"],
            "smoke_pass": (c["complete"] + c["needs_review"]) > 0}
