"""Each person's agent: reads its owner's current request, checks the other person's public profile, asks one
yes/no question at a time, and answers questions from facts it extracted from its owner's own texts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from social_agent.vocab import ScenarioSpec
from social_agent.schemas import (Answer, Assessment, Condition, ConditionAssessment, ConditionStatus, Decision,
                                  Direction, KnowledgeState, Message, MessageType, Slot, decide_direction, reconcile_decision)

# ----------------------------------------------------------------------------- what an agent may see
class AgentContext(BaseModel):
    self_id: str
    other_id: str
    self_profile: dict[str, Any]          # own agent_inputs record (self_description, request_text, evidence, policy)
    other_public_card: dict[str, Any]     # public directory entry only
    received_messages: list[Message] = Field(default_factory=list)


_ALLOWED_SELF_KEYS = {"persona_id", "name", "scenario", "self_description", "request_text", "evidence", "disclosure_policy"}
_ALLOWED_PUBLIC_KEYS = {"persona_id", "name", "scenario", "public_text"}


def build_agent_context(agent_inputs: dict[str, Any], self_id: str, other_id: str,
                        received: list[Message] | None = None) -> AgentContext:
    if self_id == other_id:
        raise ValueError("self and other must differ")
    own = agent_inputs["personas"][self_id]
    pub = agent_inputs["public_directory"][other_id]
    for m in received or []:
        if m.recipient != self_id:
            raise ValueError(f"message {m.turn} is not addressed to {self_id}")
    return AgentContext(
        self_id=self_id, other_id=other_id,
        self_profile={k: own[k] for k in own if k in _ALLOWED_SELF_KEYS},
        other_public_card={k: pub[k] for k in pub if k in _ALLOWED_PUBLIC_KEYS},
        received_messages=list(received or []),
    )


# ----------------------------------------------------------------------------- the agent
PROMPT_VERSION = "agent_v5"


# ----------------------------------------------------------------------------- LLM output schemas

class _ExtCond(BaseModel):
    attribute: str = Field(description="vocabulary attribute name, or 'other' if not in the vocabulary")
    accepted_values: list[str] = Field(description="vocabulary values the counterpart may have; empty for 'other'")
    text: str = Field(description="the requirement in one sentence")
    source_span: str = Field(description="verbatim words from the request that state it")
    required: bool = Field(description="true only if the request makes it a hard requirement")


class _Extraction(BaseModel):
    conditions: list[_ExtCond]


class _CondJudgement(BaseModel):
    condition_id: str
    status: Literal["satisfied", "conflict", "unknown"]
    evidence_quote: str = Field(description="verbatim words from the counterpart's card or answers; empty if unknown")
    rationale: str


class _AssessOut(BaseModel):
    judgements: list[_CondJudgement]
    proposed_decision: Literal["recommend", "reject", "insufficient_info"]


class _SelfFactText(BaseModel):
    attribute: str
    value: str | list[str] = Field(description="vocabulary value(s); 'unknown' if the text does not state it")
    quote: str = Field(description="verbatim words from the text that state it; empty if unknown")


class _SelfFactsText(BaseModel):
    facts: list[_SelfFactText]


class _QuestionOut(BaseModel):
    question: str


# ----------------------------------------------------------------------------- prompts

VOCAB_DESCRIPTIONS = True   # value glossary in prompts; recorded in run_meta. Off = bare value tokens only.


def vocab_block(s: ScenarioSpec) -> str:
    lines = []
    for a, sp in s.attributes.items():
        kind = "set: a person may have several" if sp.kind == "set" else "single value"
        if VOCAB_DESCRIPTIONS:
            vals = "; ".join(f"{v} = {sp.meaning.get(v) or sp.phrases.get(v, v)}" for v in sp.values)
            lines.append(f"- {a} ({kind}): {vals}")
        else:
            lines.append(f"- {a} ({kind}): {sp.values}")
    lines.append("Values are labels; each glossary entry describes the PERSON WHO HOLDS that value. Judge by meaning, not by whether the label word appears in the text.")
    return "\n".join(lines)


_SYS_EXTRACT = """You are the personal agent of one user. Read the user's CURRENT REQUEST and turn every requirement they place on a partner into a condition using the vocabulary below. Rules:
- one condition per attribute; accepted_values are the values the PARTNER must hold (not the requester's own value). Example: "I need someone who can teach me post-processing" -> the partner must be a teacher -> editing_role accepted_values ["teach"]; "I need someone who wants to learn from me" -> ["learn"];
- keep EVERY alternative the request names: "X or Y", "either X or Y", "any of X, Y" become accepted_values [X, Y]; never drop a named alternative, and never add a vocabulary value the request does not name;
- required=true only when the request states it as a must; wording like 'ideally' or 'not a deal-breaker' is required=false;
- source_span must be copied verbatim from the request;
- if a requirement does not fit any attribute, use attribute 'other' with empty accepted_values;
- never invent requirements that are not in the request.
Vocabulary:
{vocab}"""

_SYS_SELF_FACTS_TEXT = """You are the personal agent of one user. From the owner's OWN natural-language texts (self-description, current request), record the owner's value for every vocabulary attribute, with a verbatim quote from the text. Rules: use vocabulary values only; set attributes take the list of every value stated; if the text does not state an attribute, value = "unknown" and quote = ""; never infer from related statements (a session format says nothing about who explains; a hobby says nothing about pace); the request text describes what the owner wants from a PARTNER, so use it for the owner's own facts only where it states something about the owner (e.g. "this time I'm after a relaxed walk").
Vocabulary:
{vocab}"""

_SYS_ASSESS = """You are the personal agent of user {name}. For each of the owner's conditions, judge whether the COUNTERPART satisfies it using only the counterpart's public card and the answers listed. Rules:
- satisfied: the card/answers state, for THIS attribute, a value inside the accepted values;
- conflict: the card/answers state, for THIS attribute, a value that is outside ALL accepted values (e.g. accepted ["learn"] and the card says "happy to teach" -> conflict; accepted ["sun_morning"] and the card lists availability "Saturday afternoon" only -> conflict, because a listed availability/communication set is the person's complete set);
- unknown: the card/answers say nothing about THIS attribute at all. Only then. Missing information is never a conflict and never satisfied;
- a related activity, role or general statement does NOT confirm a specific condition: "tutoring session" says nothing about silence, "photo walk" says nothing about portraits;
- evidence_quote must be copied verbatim from the card or an answer and must state the value itself; the program rejects judgements whose quote is missing or not found and sets them to unknown. Leave the quote empty for unknown.
Then propose a decision: reject if any required condition is in conflict, insufficient_info if any required condition is unknown, else recommend. The program will recompute the decision; your proposal is recorded for comparison.
Vocabulary (labels with their meaning):
{vocab}"""

_SYS_QUESTION = """You are the personal agent of user {name}. Write ONE short, polite yes/no question to the counterpart's agent about the condition below. POLARITY RULE: the question must be phrased so that answering "yes" means the condition HOLDS for the counterpart (e.g. condition 'a slow or moderate pace' -> "Are you fine with a slow or moderate pace?", never "Do you prefer a fast pace?"). Mention every accepted alternative. Do not reveal anything about the owner beyond the condition itself. Do not ask about anything else."""

# ----------------------------------------------------------------------------- agent

def _norm(t: str) -> str:
    t = t.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"').replace("\u2011", "-").replace("\u2010", "-")
    return " ".join(t.lower().split())


def _quote_found(quote: str, pool: str) -> bool:
    """Verbatim check; an elided quote ("A ... B") passes only if every segment is found verbatim."""
    if not quote or not quote.strip():
        return False
    q = _norm(quote).strip().strip('"\'`“”‘’').strip()
    segs = [x.strip(" .,;\"'") for x in q.replace("…", "...").split("...")]
    segs = [x for x in segs if x]
    return bool(segs) and all(x in pool for x in segs)


def _hash(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]


class PersonalAgent:
    def __init__(self, ctx: AgentContext, scenario: ScenarioSpec, client, *, cache_dir: Optional[Path] = None,
                 cache_key_parts: tuple = ()):
        self.ctx = ctx
        self.s = scenario
        self.client = client
        self.cache_dir = cache_dir
        self.cache_key_parts = cache_key_parts
        self.conditions: list[Condition] = []
        self.assessments: dict[str, ConditionAssessment] = {}
        self.answered: dict[str, tuple[Answer, list[str], str]] = {}   # condition_id -> (answer, evidence_ids, note)
        self.usage: list[dict[str, Any]] = []
        self.extraction_raw: Optional[dict[str, Any]] = None

    # ---- helpers
    @property
    def pid(self) -> str:
        return self.ctx.self_id

    @property
    def name(self) -> str:
        return self.ctx.self_profile["name"]

    def _record(self, step: str, res) -> None:
        rec = res.to_record(); rec["step"] = step
        self.usage.append({k: rec[k] for k in ("step", "ok", "error", "calls", "network_retries", "json_repairs",
                                                 "schema_downgrades", "prompt_tokens", "completion_tokens",
                                                 "tokens_incomplete", "http_latency_s", "wall_s", "truncated")})

    def allowed_evidence_ids(self) -> set[str]:
        pol = self.ctx.self_profile["disclosure_policy"]
        return set(pol["public_evidence_ids"]) | set(pol["disclosable_evidence_ids"])

    # ---- Understand
    async def extract(self) -> list[Condition]:
        key = _hash("extract", PROMPT_VERSION, self.pid, self.ctx.self_profile["request_text"], *self.cache_key_parts)
        cf = self.cache_dir / "extract" / f"{key}.json" if self.cache_dir else None
        if cf and cf.exists():
            d = json.loads(cf.read_text(encoding="utf-8"))
            self.conditions = [Condition.model_validate(c) for c in d["conditions"]]
            self.extraction_raw = {"cache_hit": True, "key": key}
            return self.conditions
        user = f"CURRENT REQUEST: {self.ctx.self_profile['request_text']}"
        res = await self.client.structured(_SYS_EXTRACT.format(vocab=vocab_block(self.s)), user, _Extraction,
                                           request_id=f"extract:{self.pid}")
        self._record("extract", res)
        if not res.ok:
            raise AgentError(f"extract failed: {res.error}")
        req = self.ctx.self_profile["request_text"]
        conds: list[Condition] = []
        seen: set[str] = set()
        for i, c in enumerate(res.parsed.conditions):
            attr = c.attribute if c.attribute in self.s.attributes else "other"
            if attr in seen and attr != "other":
                continue
            seen.add(attr)
            vals = [v for v in c.accepted_values if attr != "other" and v in self.s.attributes[attr].values]
            confirmed = c.required and c.source_span.strip() != "" and c.source_span.strip() in req
            conds.append(Condition(id=f"{self.pid}.c{i}", text=c.text, source_span=c.source_span, slot=Slot.expect,
                                   direction=Direction.about_other, required=c.required, confirmed=confirmed,
                                   knowledge_state=KnowledgeState.known, vocab_key=None if attr == "other" else attr,
                                   value=json.dumps(vals) if vals else None))
        self.conditions = conds
        self.extraction_raw = {"cache_hit": False, "key": key, "raw": res.raw_text}
        if cf:
            cf.parent.mkdir(parents=True, exist_ok=True)
            cf.write_text(json.dumps({"conditions": [c.model_dump() for c in conds], "prompt_version": PROMPT_VERSION,
                                      "model": res.model}, indent=1), encoding="utf-8")
        return conds

    # ---- Own facts (once per persona, cached) for program-side answering
    async def extract_self_facts_from_text(self) -> dict[str, dict[str, Any]]:
        """Natural-text path: facts from the owner's own visible texts, each anchored by a verbatim quote (program-checked).
        Disclosure is decided per attribute by the owner's policy (evidence ids are per attribute)."""
        prof = self.ctx.self_profile
        texts = [("self_description", prof.get("self_description") or "")] + [("request_text", prof.get("request_text") or "")]
        for extra in prof.get("extra_request_texts", []):
            texts.append(("request_text", extra))
        key = _hash("selffacts_text", PROMPT_VERSION, self.pid, [t for _, t in texts], *self.cache_key_parts)
        cf = self.cache_dir / "selffacts_text" / f"{key}.json" if self.cache_dir else None
        if cf and cf.exists():
            self.self_facts = json.loads(cf.read_text(encoding="utf-8"))["facts"]
            return self.self_facts
        user = "\n\n".join(f"[{name}]\n{t}" for name, t in texts if t)
        res = await self.client.structured(_SYS_SELF_FACTS_TEXT.format(vocab=vocab_block(self.s)), user, _SelfFactsText,
                                           request_id=f"selffacts_text:{self.pid}")
        self._record("selffacts_text", res)
        if not res.ok:
            raise AgentError(f"self-fact (text) extraction failed: {res.error}")
        pool = _norm(" ".join(t for _, t in texts))
        allowed_attrs = {i.split(".", 1)[1] for i in self.allowed_evidence_ids() if "." in i}
        facts: dict[str, dict[str, Any]] = {}
        for f in res.parsed.facts:
            if f.attribute not in self.s.attributes:
                continue
            sp = self.s.attributes[f.attribute]; v = f.value
            if sp.kind == "set":
                vals = [x for x in (v if isinstance(v, list) else [v]) if x in sp.values]; v = vals if vals else "unknown"
            else:
                v = v if (isinstance(v, str) and v in sp.values) else "unknown"
            grounded = v != "unknown" and _quote_found(f.quote, pool)
            if v != "unknown" and not grounded:
                v = "unknown"                                   # a fact the owner's text does not literally state is not known
            facts[f.attribute] = {"value": v, "quote": f.quote if grounded else "",
                                  "evidence_id": f"{self.pid}.{f.attribute}" if (grounded and f.attribute in allowed_attrs) else "",
                                  "disclosable": f.attribute in allowed_attrs}
        self.self_facts = facts
        if cf:
            cf.parent.mkdir(parents=True, exist_ok=True)
            cf.write_text(json.dumps({"facts": facts, "prompt_version": PROMPT_VERSION, "model": res.model}, indent=1), encoding="utf-8")
        return facts

    def answer_from_facts(self, ask: Message, turn: int) -> Message:
        """Program answer: own extracted fact vs the asked accepted values; evidence filtered by disclosure policy."""
        attr = ask.note.split("attribute ", 1)[-1].split(" is one of", 1)[0] if ask.note else ""
        try:
            accepted = json.loads(ask.note.split(" is one of ", 1)[1].split("; in words", 1)[0])
        except Exception:
            accepted = []
        f = getattr(self, "self_facts", {}).get(attr)
        ans, ev, note = Answer.unknown, [], "no stated fact for this attribute"
        if f and f["value"] != "unknown" and attr in self.s.attributes:
            if f["evidence_id"] and f["evidence_id"] in self.allowed_evidence_ids():
                v = f["value"]
                hit = bool(set(v) & set(accepted)) if isinstance(v, list) else v in accepted
                ans, ev, note = (Answer.yes if hit else Answer.no), [f["evidence_id"]], "program answer from stated fact"
            else:
                note = "fact exists but is not disclosable"
        return Message(session_id=ask.session_id, turn=turn, sender=self.pid, recipient=ask.sender, type=MessageType.ANSWER,
                       condition_id=ask.condition_id, answer=ans, evidence_ids=ev, note=note)

    # ---- Match: directional assessment
    def _answers_block(self) -> str:
        if not self.answered:
            return "(none)"
        by = {c.id: c for c in self.conditions}
        return "\n".join(f"- about '{by[cid].text}': counterpart answered {a.value}; note: {note}"
                         for cid, (a, _, note) in self.answered.items())

    async def assess(self, step: str = "assess") -> Assessment:
        cond_lines = "\n".join(f"- id={c.id} [{'MUST' if c.enforceable else 'prefer'}] attribute={c.vocab_key or 'other'} "
                               f"accepted={c.value or '[]'}: {c.text}" for c in self.conditions)
        user = (f"OWNER'S CONDITIONS:\n{cond_lines}\n\nCOUNTERPART PUBLIC CARD:\n{self.ctx.other_public_card['public_text']}\n\n"
                f"ANSWERS RECEIVED FROM THE COUNTERPART'S AGENT:\n{self._answers_block()}")
        res = await self.client.structured(_SYS_ASSESS.format(name=self.name, vocab=vocab_block(self.s)), user, _AssessOut,
                                           request_id=f"{step}:{self.pid}")
        self._record(step, res)
        if not res.ok:
            raise AgentError(f"{step} failed: {res.error}")
        valid = {c.id for c in self.conditions}
        pool = _norm(self.ctx.other_public_card["public_text"] + " " + " ".join(n for _, _, n in self.answered.values()))
        judged: dict[str, ConditionAssessment] = {}
        for j in res.parsed.judgements:
            if j.condition_id in valid and j.condition_id not in judged:
                st = ConditionStatus(j.status)
                ca = ConditionAssessment(condition_id=j.condition_id, status=st, llm_status=st, rationale=j.rationale,
                                         evidence_quote=j.evidence_quote)
                # evidence guard: satisfied/conflict need a verbatim quote from visible, authorized text
                if st != ConditionStatus.unknown and not _quote_found(j.evidence_quote, pool):
                    ca.status, ca.guard = ConditionStatus.unknown, "no_verbatim_evidence"
                judged[j.condition_id] = ca
        for c in self.conditions:               # anything the model skipped is unknown, never satisfied
            judged.setdefault(c.id, ConditionAssessment(condition_id=c.id, status=ConditionStatus.unknown, rationale="not judged", guard="not_judged"))
        self._apply_answers(judged)
        self.assessments = judged
        final, ok = reconcile_decision(Decision(res.parsed.proposed_decision), list(judged.values()), self.conditions)
        self._last_proposed = Decision(res.parsed.proposed_decision)
        return Assessment(assessor=self.pid, target=self.ctx.other_id, conditions=list(judged.values()), decision=final,
                          unresolved=[c.id for c in self.conditions if c.enforceable and judged[c.id].status == ConditionStatus.unknown],
                          proposed=Decision(res.parsed.proposed_decision), proposal_matches=ok)

    def _apply_answers(self, judged: dict[str, ConditionAssessment]) -> None:
        """Program-side update from handshake answers: only the asked condition changes, by the bound polarity."""
        for cid, (a, ev, _) in self.answered.items():
            if cid not in judged:
                continue
            prev = judged[cid]
            if a == Answer.yes:
                judged[cid] = ConditionAssessment(condition_id=cid, status=ConditionStatus.satisfied, evidence_ids=ev,
                                                  rationale="answered yes", llm_status=prev.llm_status, guard="answered_yes")
            elif a == Answer.no:
                judged[cid] = ConditionAssessment(condition_id=cid, status=ConditionStatus.conflict, evidence_ids=ev,
                                                  rationale="answered no", llm_status=prev.llm_status, guard="answered_no")
            else:
                judged[cid] = ConditionAssessment(condition_id=cid, status=ConditionStatus.unknown, evidence_ids=[],
                                                  rationale="answered unknown", llm_status=prev.llm_status, guard="answered_unknown")

    def update_from_answers(self) -> Assessment:
        """Re-derive the directional assessment from stored condition states + answers, with NO LLM call."""
        judged = dict(self.assessments)
        self._apply_answers(judged)
        self.assessments = judged
        final = decide_direction(list(judged.values()), self.conditions)
        prop = getattr(self, "_last_proposed", None)
        return Assessment(assessor=self.pid, target=self.ctx.other_id, conditions=list(judged.values()), decision=final,
                          unresolved=[c.id for c in self.conditions if c.enforceable and judged[c.id].status == ConditionStatus.unknown],
                          proposed=prop, proposal_matches=(prop == final) if prop else None)

    # ---- Ask
    def next_unresolved(self, asked: set[str]) -> Optional[Condition]:
        for c in self.conditions:               # program picks; fixed order = condition order
            if c.enforceable and c.id not in asked and c.id not in self.answered \
                    and self.assessments.get(c.id, ConditionAssessment(condition_id=c.id, status=ConditionStatus.unknown)).status == ConditionStatus.unknown:
                return c
        return None

    async def ask(self, cond: Condition, session_id: str, turn: int) -> Message:
        user = f"CONDITION: {cond.text} (attribute {cond.vocab_key or 'other'}, accepted values {cond.value or 'n/a'})"
        res = await self.client.structured(_SYS_QUESTION.format(name=self.name), user, _QuestionOut, request_id=f"ask:{self.pid}:{cond.id}")
        self._record("ask", res)
        if not res.ok:
            raise AgentError(f"ask failed: {res.error}")
        polarity = f"CONDITION (answer yes only if this holds for you): attribute {cond.vocab_key or 'other'} is one of {cond.value or 'n/a'}; in words: {cond.text}"
        return Message(session_id=session_id, turn=turn, sender=self.pid, recipient=self.ctx.other_id, type=MessageType.ASK,
                       condition_id=cond.id, question=res.parsed.question, note=polarity)

    # ---- Answer (outbound check by the program)
    def receive_answer(self, m: Message) -> None:
        self.ctx.received_messages.append(m)
        self.answered[m.condition_id] = (m.answer, list(m.evidence_ids), m.note or "")


class AgentError(RuntimeError):
    pass
