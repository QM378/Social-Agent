"""Community dataset v2 (paper experiment material, not a general platform).

Per person: STABLE facts (capabilities, objective availability, habitual preferences) with visibility.
Per request: a CURRENT STATE = stable facts + at most one situational override, plus requirements on the partner
  (1-3 required for normal requests, 4 for ~10% strict ones; >=1 preference with weight 1 or 2).
Requirement sources (guides, not quotas): derived from own state, independent partner requirement, situational.
Hard rules checked in code (check_consistency):
  * a requirement never derives from a withheld fact;
  * role requirements respect capability (only a teacher may require a learner, only a learner may require a teacher);
  * availability / venue / communication requirements are the owner's own set (feasibility);
  * a situational override may only relax willingness (teach->none, fast->slow) or change the activity type;
    it never changes availability or invents a capability.
Alternative requests (20%): exactly one factor changes (activity type | one required condition relaxed to a preference |
  willingness). Community state for a switch: only that person changes; the program stores the NEW ROW (their new
  request vs everyone) and the UPDATED COLUMN (everyone's primary request vs their new state), under the switch id.
Labels: full-fact score (3u | 4+5u) for ranking; visible three-class status (public view, disclosable view).
Demographics are narrative only; a test asserts they do not affect any matrix.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from social_agent.vocab import Fact, Persona, Requirement, ScenarioSpec, Vocab, _NAMES, STYLES, eval_requirement
from social_agent.schemas import ConditionStatus, Decision, Visibility

AGE_BANDS = ["early 20s", "mid 20s", "late 20s", "early 30s", "mid 30s", "late 30s", "40s", "50s"]
GENDERS = ["woman", "man", "non-binary", "prefer not to say"]
NATIONALITIES = ["American", "Mexican", "Canadian", "Brazilian", "German", "Nigerian", "Indian", "Japanese", "Korean",
                 "Turkish", "Spanish", "Vietnamese", "Egyptian", "Australian", "Polish", "Kenyan", "Italian", "Chilean"]
HOBBIES = ["baking sourdough", "running trails", "collecting vinyl", "learning to sail", "growing chillies", "watching old westerns",
           "playing chess online", "sketching buildings", "birdwatching", "knitting", "restoring bikes", "amateur astronomy",
           "podcasting", "urban gardening", "swing dancing", "climbing", "fermenting hot sauce", "reading history", "woodworking"]
ACTIVITY_ATTR = {"photography": "purpose", "gaming": "mode", "study": "format"}
ROLE_ATTR = {"photography": "editing_role", "gaming": "coaching_role", "study": "tutor_role"}
FEASIBILITY_ATTRS = {"time_window", "venue", "comm_modes"}
WILLINGNESS_RELAX = {"editing_role": {"teach": "none"}, "coaching_role": {"teach": "none"}, "tutor_role": {"teach": "none"},
                     "pace": {"fast": "slow", "moderate": "slow"}}
LENGTH_BANDS = {"short": (40, 80), "medium": (80, 160), "long": (160, 250)}
# ----------------------------------------------------------------------------- template sentences
# Deterministic sentences from the vocabulary phrases. They give the generation notes for the LLM writers and the public
# profile cards (public facts only). The LLM-written self-descriptions and requests are produced in community/texts.py.
_TIME = {"sat_morning", "sat_afternoon", "sun_morning", "sun_afternoon", "weekday_evening"}


class RenderedPersona(BaseModel):
    persona_id: str
    mode: str                                   # template | llm
    fact_sentences: list[dict[str, str]]        # {evidence_id, attribute, visibility, text}
    self_description: str                       # what the owner's agent sees (public + disclosable facts)
    public_text: str                            # discovery card text (public facts only)
    request_text: str                           # current request (explicit required conditions + preferences)
    sources: dict[str, str] = Field(default_factory=dict)   # per text: template | llm
    generator: dict[str, Any] = Field(default_factory=dict)


def _phrase(s: ScenarioSpec, attr: str, value: str | list[str]) -> str:
    sp = s.attributes[attr]
    if isinstance(value, list):
        ps = [sp.phrases[v] for v in value]
        return ps[0] if len(ps) == 1 else ", ".join(ps[:-1]) + " or " + ps[-1]
    return sp.phrases[value]


_FACT_TPL = {
    "plain": {
        "purpose": "What I am after is {p}.", "pace": "I like {p}.", "gear_talk": "On equipment: {p}.",
        "editing_role": "For post-processing, {p}.", "mutual_portraits": "I am fine with {p}.",
        "time_window": "I am free {p}.",
        "game_id": "I mainly play {p}.", "mode": "I usually queue for {p}.", "comm_modes": "I communicate over {p}.",
        "team_role": "I play {p}.", "coaching_role": "About coaching: {p}.",
        "topic": "I am studying {p}.", "format": "I work best with {p}.", "quiet": "I need {p}.",
        "tutor_role": "On tutoring: {p}.", "venue": "I can meet {p}.",
    },
    "chatty": {
        "purpose": "Honestly, I'm just looking for {p}.", "pace": "I tend to go at {p}, that's how I enjoy it.",
        "gear_talk": "Fair warning, {p} is my thing.", "editing_role": "As for editing, {p}.",
        "mutual_portraits": "I'm cool with {p}.", "time_window": "Most weeks I can do {p}.",
        "game_id": "These days it's all {p} for me.", "mode": "I'm mostly into {p}.", "comm_modes": "I'm on {p}.",
        "team_role": "I usually end up in {p}.", "coaching_role": "Coaching-wise, {p}.",
        "topic": "Right now it's {p} for me.", "format": "I get the most out of {p}.", "quiet": "I do best with {p}.",
        "tutor_role": "Tutoring-wise, {p}.", "venue": "I'm happy {p}.",
    },
    "terse": {
        "purpose": "Looking for {p}.", "pace": "{P}.", "gear_talk": "{P}.", "editing_role": "{P}.",
        "mutual_portraits": "{P}.", "time_window": "Available {p}.",
        "game_id": "{P}.", "mode": "{P}.", "comm_modes": "{P} only.", "team_role": "{P}.", "coaching_role": "{P}.",
        "topic": "{P}.", "format": "{P}.", "quiet": "{P}.", "tutor_role": "{P}.", "venue": "{P}.",
    },
}
_BG_TPL = {
    "camera_brand": "I shoot with {v}.", "platform": "I play on {v}.", "field_of_study": "My major is {v}.",
}
_ACTIVITY = {"photography": "a photography partner", "gaming": "a gaming partner", "study": "a study partner"}


def _cap(t: str) -> str:
    return t[:1].upper() + t[1:]


def fact_sentence(s: ScenarioSpec, style: str, f: Fact) -> str:
    p = _phrase(s, f.attribute, f.value)
    tpl = _FACT_TPL[style].get(f.attribute, "{P}.")
    return tpl.format(p=p, P=_cap(p))


_REQ_PHRASES = {
    "editing_role": {"teach": "you to be the one who teaches me post-processing", "learn": "you to be the one who learns post-processing from me"},
    "coaching_role": {"teach": "you to be the one who coaches me", "learn": "you to be the one who gets coached by me"},
    "tutor_role": {"teach": "you to be the one who explains the material to me", "learn": "you to be the one who has things explained by me"},
    "team_role": {"support": "you to play a support role", "damage": "you to play a damage role", "flexible": "you to be flexible about your role"},
}


def req_sentence(s: ScenarioSpec, r: Requirement) -> str:
    sp = s.attributes[r.attribute]
    vals = r.accepted_values
    if r.attribute in _REQ_PHRASES and all(v in _REQ_PHRASES[r.attribute] for v in vals):
        ps = [_REQ_PHRASES[r.attribute][v] for v in vals]
        core = "I need " + (ps[0] if len(ps) == 1 else ", ".join(ps[:-1]) + " or " + ps[-1])
        return _cap(core) + ". This is a must for me." if r.required else "Ideally " + core + ", but that is not a deal-breaker."
    if r.operator == "intersects":
        p = _phrase(s, r.attribute, vals)
        body = f"any of {p}" if len(vals) > 1 else p
        core = f"you need to be available {body}" if r.attribute == "time_window" else f"we need to overlap on {body}"
    else:
        ps = [sp.phrases[v] for v in vals]
        body = ps[0] if len(ps) == 1 else ", ".join(ps[:-1]) + " or " + ps[-1]
        core = f"I need someone who is into {body}" if r.attribute in ("purpose", "game_id", "mode", "topic", "format") \
            else f"I need {body} from you" if r.attribute.endswith("_role") \
            else f"you should be okay with {body}"
    if r.required:
        return _cap(core) + ". This is a must for me."
    return "Ideally " + core + ", but that is not a deal-breaker."


def render_template(vocab: Vocab, p: Persona) -> RenderedPersona:
    s = vocab.scenarios[p.scenario]
    sents = []
    for a in s.attributes:            # fixed attribute order; style varies wording only
        f = p.facts[a]
        sents.append({"evidence_id": f.evidence_id, "attribute": a, "visibility": f.visibility.value,
                      "text": fact_sentence(s, p.style, f)})
    bg = " ".join(_BG_TPL[k].format(v=v) for k, v in p.background.items())
    own_visible = [x["text"] for x in sents if x["visibility"] != Visibility.withheld.value]
    public = [x["text"] for x in sents if x["visibility"] == Visibility.public.value]
    intro = f"Hi, I'm {p.name}. " if p.style != "terse" else f"{p.name}. "
    self_desc = intro + bg + " " + " ".join(own_visible)
    public_text = f"{p.name}, {_ACTIVITY[p.scenario]}. " + " ".join(public)
    req = " ".join(req_sentence(s, r) for r in p.requirements)
    request = f"I'm looking for {_ACTIVITY[p.scenario]}. " + req
    return RenderedPersona(persona_id=p.persona_id, mode="template", fact_sentences=sents,
                           self_description=self_desc.strip(), public_text=public_text.strip(),
                           request_text=request.strip(),
                           sources={"self_description": "template", "public_text": "template", "request_text": "template"},
                           generator={"mode": "template", "vocab_version": vocab.version})


TEXT_STYLES = ["concise", "conversational", "narrative"]

_EXTRA_NAMES = ["Amara", "Bastian", "Celia", "Dmitri", "Elif", "Farah", "Gideon", "Hiro", "Imani", "Jonas", "Kenji", "Leila",
                "Mateo", "Nadia", "Oren", "Priya", "Rafael", "Sana", "Tomas", "Ulla", "Viktor", "Wanjiru", "Yara", "Zara",
                "Aiko", "Bruno", "Chiara", "Emeka", "Freya", "Gabriel", "Isaac", "Jamal", "Katya", "Luca", "Maya", "Nikolai",
                "Olu", "Paola", "Ravi", "Sofia", "Tariq", "Vera", "Wei", "Ximena", "Yusuf", "Zoe", "Anouk", "Bilal", "Carmen",
                "Diego", "Esther", "Greta", "Hamza", "Kofi", "Lena", "Marco", "Noa", "Omar", "Petra", "Rosa", "Samir", "Thea",
                "Umar", "Vanya", "Willa", "Zain", "Alina", "Boris", "Clara", "Elena", "Faisal", "Gia", "Henrik", "Ida",
                "Joaquin", "Kaia", "Milo", "Nour", "Otto", "Pia", "Reza", "Selin", "Tove", "Ugo", "Vida", "Yuki", "Zeynep",
                "Adaeze", "Benji", "Cora", "Darius", "Edda", "Femi", "Hakim", "Iris", "Javi", "Kenna", "Lars", "Malik",
                "Odalys", "Pavel", "Rhea", "Sven", "Tamsin", "Uriel", "Vince", "Wiktor", "Yolanda", "Ziad", "Aditi", "Brent",
                "Camila", "Dorian", "Eloise", "Fabio", "Gloria", "Hugo", "Irene", "Jaya", "Klaus", "Lucia", "Matteo", "Nina",
                "Oskar", "Polina", "Quentin", "Rania", "Stefan", "Talia", "Usman", "Valeria", "Wolfgang", "Yasmin", "Zoltan"]
NAME_POOL = list(dict.fromkeys(_NAMES + _EXTRA_NAMES))


class RequestSpec(BaseModel):
    request_id: str
    person_id: str
    kind: str                                   # primary | alternative
    strict: bool = False
    change: Optional[dict[str, Any]] = None
    overrides: dict[str, Any] = Field(default_factory=dict)
    requirements: list[Requirement]
    weights: dict[str, float] = Field(default_factory=dict)
    sources: dict[str, str] = Field(default_factory=dict)


class CommunityPerson(BaseModel):
    person_id: str
    name: str
    age_band: str
    gender: str
    nationality: str
    hobby: str
    scenario: str
    style: str
    text_style: str
    length_band: str
    facts: dict[str, Fact]
    background: dict[str, str] = Field(default_factory=dict)


def _pick(rng: random.Random, spec):
    if spec.kind == "enum":
        return rng.choice(spec.values)
    k = rng.randint(1, min(3, len(spec.values)))
    return sorted(rng.sample(spec.values, k))


def current_state(p: CommunityPerson, r: RequestSpec) -> dict[str, Any]:
    st = {a: f.value for a, f in p.facts.items()}
    st.update(r.overrides)
    return st


def _derived(s: ScenarioSpec, rid: str, attr: str, own, required: bool) -> Optional[Requirement]:
    sp = s.attributes[attr]
    if sp.kind == "set":
        if set(own) >= set(sp.values):
            return None
        acc, op = list(own), "intersects"
    else:
        if not sp.accepts or sp.accepts.get(own) is None or set(sp.accepts[own]) >= set(sp.values):
            return None
        acc, op = list(sp.accepts[own]), "in"
    return Requirement(id=f"{rid}.req.{attr}", attribute=attr, operator=op, accepted_values=acc, required=required,
                       confirmed=required, source="request" if required else "preference")


def _independent(rng: random.Random, s: ScenarioSpec, rid: str, attr: str, required: bool) -> Optional[Requirement]:
    sp = s.attributes[attr]
    if attr in FEASIBILITY_ATTRS or attr in ROLE_ATTR.values() or sp.kind == "set":
        return None
    k = rng.randint(1, max(1, len(sp.values) - 1))
    acc = sorted(rng.sample(sp.values, k))
    return Requirement(id=f"{rid}.req.{attr}", attribute=attr, operator="in", accepted_values=acc, required=required,
                       confirmed=required, source="request" if required else "preference")


def _role_ok(state: dict[str, Any], role_attr: str, req: Requirement) -> bool:
    if req.attribute != role_attr:
        return True
    own = state[role_attr]
    if own == "teach":
        return req.accepted_values == ["learn"]
    if own == "learn":
        return req.accepted_values == ["teach"]
    return False


def _make_requirements(rng: random.Random, s: ScenarioSpec, p: CommunityPerson, rid: str, state: dict[str, Any],
                       strict: bool, withheld: set[str], p_independent: float = 0.35):
    """>=1 preference reserved first, then 1-3 (or 4 if strict) required conditions. Returns ([], {}, {}) if impossible."""
    role_attr = ROLE_ATTR[p.scenario]

    def can_independent(a: str) -> bool:
        return a not in FEASIBILITY_ATTRS and a != role_attr and s.attributes[a].kind == "enum"

    def can_derive(a: str) -> bool:
        q = _derived(s, rid, a, state[a], True)
        return q is not None and _role_ok(state, role_attr, q)

    def build(a: str, required: bool):
        use_ind = can_independent(a) and (rng.random() < p_independent or not can_derive(a))
        if use_ind:
            return _independent(rng, s, rid, a, required), "independent"
        return _derived(s, rid, a, state[a], required), "derived"

    requirable = [a for a in s.attributes if a not in withheld and (can_derive(a) or can_independent(a))]
    if len(requirable) < 2:
        return [], {}, {}
    rng.shuffle(requirable)
    n_pref = rng.randint(1, 3)
    n_req = 4 if strict else rng.randint(1, 3)
    n_pref = max(1, min(n_pref, len(requirable) - 1))
    n_req = max(1, min(n_req, len(requirable) - n_pref))
    pref_attrs, req_attrs = requirable[:n_pref], requirable[n_pref:n_pref + n_req]
    reqs, sources, weights = [], {}, {}
    for a in req_attrs:
        q, src = build(a, True)
        if q:
            reqs.append(q); sources[q.id] = src
    for a in pref_attrs:
        q, src = build(a, False)
        if q:
            reqs.append(q); sources[q.id] = src; weights[q.id] = rng.choice([1.0, 1.0, 2.0])
    return reqs, sources, weights


def _rename(q: Requirement, rid: str) -> Requirement:
    return q.model_copy(update={"id": f"{rid}.req.{q.attribute}"})


def _linked_rederive(s: ScenarioSpec, rid: str, q: Requirement, new_state: dict[str, Any], src: str) -> Optional[Requirement]:
    """A requirement on the changed attribute that was DERIVED from the owner's own value follows the new value.
    Independent requirements and requirements on other attributes are untouched."""
    if src != "derived":
        return _rename(q, rid)
    return _derived(s, rid, q.attribute, new_state[q.attribute], q.required)


def _copy_with_change(s: ScenarioSpec, p: CommunityPerson, primary: RequestSpec, rid: str, changed_attr: str,
                      new_state: dict[str, Any]) -> tuple[list[Requirement], dict, dict, list[str]]:
    """Copy the primary request; only conditions on `changed_attr` (and role conditions that lose their capability) change."""
    role_attr = ROLE_ATTR[p.scenario]
    reqs, weights, sources, linked = [], {}, {}, []
    for q in primary.requirements:
        src = primary.sources.get(q.id, "derived")
        if q.attribute == changed_attr:
            q2 = _linked_rederive(s, rid, q, new_state, src)
            linked.append(f"{q.attribute}: re-derived from new own value" if q2 else f"{q.attribute}: dropped (no longer applicable)")
        elif q.attribute == role_attr and not _role_ok(new_state, role_attr, q):
            q2 = None; linked.append(f"{q.attribute}: dropped (capability changed)")
        else:
            q2 = _rename(q, rid)
        if q2 is None:
            continue
        if not q2.required:
            weights[q2.id] = primary.weights.get(q.id, 1.0)
        sources[q2.id] = src; reqs.append(q2)
    return reqs, weights, sources, linked


def _alternative(rng: random.Random, s: ScenarioSpec, p: CommunityPerson, primary: RequestSpec, withheld: set[str]) -> Optional[RequestSpec]:
    """Exactly one factor changes; everything else is copied from the primary request (conditions, weights, sources)."""
    rid = f"{p.person_id}.r2"
    base = current_state(p, primary)
    options = ["activity", "relax", "willingness"]
    rng.shuffle(options)
    for factor in options:
        if factor == "relax":
            req_q = [q for q in primary.requirements if q.required]
            if len(req_q) < 2:
                continue
            drop = rng.choice(req_q)
            reqs, weights, sources = [], {}, {}
            for q in primary.requirements:
                q2 = _rename(q, rid)
                if q.id == drop.id:
                    q2 = q2.model_copy(update={"required": False, "confirmed": False, "source": "preference"}); weights[q2.id] = 2.0
                elif not q.required:
                    weights[q2.id] = primary.weights.get(q.id, 1.0)
                sources[q2.id] = primary.sources.get(q.id, "derived"); reqs.append(q2)
            return RequestSpec(request_id=rid, person_id=p.person_id, kind="alternative", strict=primary.strict,
                               change={"factor": "relax", "attribute": drop.attribute, "was": "required", "now": "preference", "linked": []},
                               overrides={}, requirements=reqs, weights=weights, sources=sources)
        if factor == "activity":
            act = ACTIVITY_ATTR[p.scenario]
            if act in withheld:
                continue
            new = rng.choice([v for v in s.attributes[act].values if v != base[act]])
            state = {**base, act: new}
            reqs, weights, sources, linked = _copy_with_change(s, p, primary, rid, act, state)
        else:
            cands = [(a, m[base[a]]) for a, m in WILLINGNESS_RELAX.items() if a in base and base[a] in m and a not in withheld]
            if not cands:
                continue
            act, new = rng.choice(cands)
            state = {**base, act: new}
            reqs, weights, sources, linked = _copy_with_change(s, p, primary, rid, act, state)
        if not any(q.required for q in reqs) or not any(not q.required for q in reqs):
            continue
        return RequestSpec(request_id=rid, person_id=p.person_id, kind="alternative", strict=primary.strict,
                           change={"factor": factor, "attribute": act, "from": base[act], "to": new, "linked": linked},
                           overrides={act: new}, requirements=reqs, weights=weights, sources=sources)
    return None


def generate_community(vocab: Vocab, seed: int, n_people: int = 20, alt_ratio: float = 0.2, strict_ratio: float = 0.10,
                       withheld_prob: float = 0.3) -> dict[str, Any]:
    rng = random.Random(seed)
    scens = list(vocab.scenarios)
    scen_list = [scens[i % len(scens)] for i in range(n_people)]
    rng.shuffle(scen_list)
    names = list(NAME_POOL); rng.shuffle(names)
    if n_people > len(names):
        raise ValueError(f"name pool has {len(names)} names; n_people={n_people}")
    people: list[CommunityPerson] = []
    seen: set[str] = set()
    for i, sc in enumerate(scen_list):
        s = vocab.scenarios[sc]; pid = f"u{i + 1:04d}"
        for _ in range(100):
            facts = {a: Fact(attribute=a, value=_pick(rng, sp), visibility=sp.default_visibility, evidence_id=f"{pid}.{a}")
                     for a, sp in s.attributes.items()}
            key = sc + json.dumps({a: f.value for a, f in facts.items()}, sort_keys=True)
            if key not in seen:
                seen.add(key); break
        people.append(CommunityPerson(person_id=pid, name=names[i], age_band=rng.choice(AGE_BANDS), gender=rng.choice(GENDERS),
                                      nationality=rng.choice(NATIONALITIES), hobby=rng.choice(HOBBIES), scenario=sc,
                                      style=rng.choice(STYLES), text_style=rng.choice(TEXT_STYLES),
                                      length_band=rng.choice(list(LENGTH_BANDS)), facts=facts,
                                      background={k: rng.choice(v) for k, v in s.background_values.items()}))
    strict_ids = set(rng.sample([p.person_id for p in people], round(strict_ratio * n_people)))
    alt_ids = set(rng.sample([p.person_id for p in people], round(alt_ratio * n_people)))
    requests: list[RequestSpec] = []
    for p in people:
        s = vocab.scenarios[p.scenario]
        withheld = {a for a, f in p.facts.items() if f.visibility == Visibility.disclosable and rng.random() < withheld_prob}
        for a in withheld:
            p.facts[a].visibility = Visibility.withheld
        state = {a: f.value for a, f in p.facts.items()}
        for attempt in range(60):
            reqs, sources, weights = _make_requirements(rng, s, p, f"{p.person_id}.r1", state, p.person_id in strict_ids, withheld)
            if any(q.required for q in reqs) and any(not q.required for q in reqs):
                break
            if attempt % 10 == 9 and withheld:          # too little to ask for: un-withhold one fact and retry
                a = withheld.pop(); p.facts[a].visibility = Visibility.disclosable
        else:
            raise RuntimeError(f"{p.person_id}: cannot build a valid primary request")
        r1 = RequestSpec(request_id=f"{p.person_id}.r1", person_id=p.person_id, kind="primary", strict=p.person_id in strict_ids,
                         requirements=reqs, weights=weights, sources=sources)
        requests.append(r1)
        if p.person_id in alt_ids:
            r2 = _alternative(rng, s, p, r1, withheld)
            if r2:
                requests.append(r2)
    check_consistency(vocab, people, requests)
    matrices = compute_matrices(vocab, people, requests)
    report = community_report(vocab, people, requests, matrices)
    return {"vocab_version": vocab.version, "vocab_hash": vocab.source_hash, "seed": seed, "n_people": n_people,
            "people": [p.model_dump(mode="json") for p in people], "requests": [r.model_dump(mode="json") for r in requests],
            "matrices": matrices, "report": report}


def check_consistency(vocab: Vocab, people: list[CommunityPerson], requests: list[RequestSpec]) -> None:
    by = {p.person_id: p for p in people}
    for r in requests:
        p = by[r.person_id]; st = current_state(p, r)
        for q in r.requirements:
            if p.facts[q.attribute].visibility == Visibility.withheld:
                raise ValueError(f"{r.request_id}: requirement on withheld fact {q.attribute}")
            if q.attribute in FEASIBILITY_ATTRS and sorted(q.accepted_values) != sorted(st[q.attribute]):
                raise ValueError(f"{r.request_id}: feasibility requirement {q.attribute} differs from own set")
            if not _role_ok(st, ROLE_ATTR[p.scenario], q):
                raise ValueError(f"{r.request_id}: role requirement contradicts capability")
        if not any(q.required for q in r.requirements) or not any(not q.required for q in r.requirements):
            raise ValueError(f"{r.request_id}: needs >=1 required and >=1 preference")
        for a, v in r.overrides.items():
            if a in FEASIBILITY_ATTRS:
                raise ValueError(f"{r.request_id}: override on availability/venue/communication")
            base = p.facts[a].value
            if not (a == ACTIVITY_ATTR[p.scenario] or WILLINGNESS_RELAX.get(a, {}).get(base) == v):
                raise ValueError(f"{r.request_id}: override {a} {base}->{v} invents a capability")


# ----------------------------------------------------------------------------- matrices

def _view(p: CommunityPerson, state: dict[str, Any], level: str) -> dict[str, Any]:
    out = {}
    for a, f in p.facts.items():
        ok = level == "full" or f.visibility == Visibility.public or (level == "disclosable" and f.visibility == Visibility.disclosable)
        out[a] = state[a] if ok else None
    return out


def _cell(req: RequestSpec, target_vals: dict[str, Any]) -> dict[str, Any]:
    st = {q.id: eval_requirement(q, target_vals.get(q.attribute)) for q in req.requirements}
    required = [q for q in req.requirements if q.required]; prefs = [q for q in req.requirements if not q.required]
    conflict = any(st[q.id] == ConditionStatus.conflict for q in required)
    unknown = any(st[q.id] == ConditionStatus.unknown for q in required)
    status = Decision.reject if conflict else Decision.insufficient_info if unknown else Decision.recommend
    wsum = sum(req.weights.get(q.id, 1.0) for q in prefs) or 1.0
    u = sum(req.weights.get(q.id, 1.0) for q in prefs if st[q.id] == ConditionStatus.satisfied) / wsum
    return {"statuses": {k: v.value for k, v in st.items()}, "required_conflict": conflict, "status": status.value, "u": round(u, 4)}


def compute_matrices(vocab: Vocab, people: list[CommunityPerson], requests: list[RequestSpec]) -> dict[str, Any]:
    by = {p.person_id: p for p in people}
    ids = [p.person_id for p in people]
    primary = {r.person_id: r for r in requests if r.kind == "primary"}
    base_state = {pid: current_state(by[pid], primary[pid]) for pid in ids}

    def cell_for(req: RequestSpec, j: str, states: dict[str, dict[str, Any]]) -> dict[str, Any]:
        me, o = by[req.person_id], by[j]
        if o.scenario != me.scenario:
            return {"score": 0.0, "required_conflict": True, "status_public": "reject", "status_disclosable": "reject", "different_activity": True}
        full = _cell(req, _view(o, states[j], "full"))
        return {"score": round(3 * full["u"] if full["required_conflict"] else 4 + 5 * full["u"], 3),
                "required_conflict": full["required_conflict"],
                "status_public": _cell(req, _view(o, states[j], "public"))["status"],
                "status_disclosable": _cell(req, _view(o, states[j], "disclosable"))["status"],
                "statuses": full["statuses"], "u": full["u"]}

    main = {pid: {j: cell_for(primary[pid], j, base_state) for j in ids if j != pid} for pid in ids}
    switches = {}
    for r in requests:
        if r.kind != "alternative":
            continue
        states = dict(base_state); states[r.person_id] = current_state(by[r.person_id], r)
        new_row = {j: cell_for(r, j, states) for j in ids if j != r.person_id}
        new_col = {j: cell_for(primary[j], r.person_id, states) for j in ids if j != r.person_id and by[j].scenario == by[r.person_id].scenario}
        switches[r.request_id] = {"person_id": r.person_id, "change": r.change, "row": new_row, "column": new_col}
    return {"person_ids": ids, "main": main, "switches": switches,
            "rule": "score = 3*u if any required conflict else 4+5*u (u = weighted satisfied share of preferences, full facts); "
                    "status_* = recommend/reject/insufficient_info from the target's visible facts"}


def community_report(vocab: Vocab, people: list[CommunityPerson], requests: list[RequestSpec], m: dict[str, Any]) -> dict[str, Any]:
    by = {p.person_id: p for p in people}; ids = m["person_ids"]; main = m["main"]
    primary = {r.person_id: r for r in requests if r.kind == "primary"}
    high = Counter(); none_ok = 0; mutual = one_way = same = 0
    for i in ids:
        cells = {j: c for j, c in main[i].items() if not c.get("different_activity")}
        high[sum(1 for c in cells.values() if c["score"] >= 7)] += 1
        if cells and not any(not c["required_conflict"] for c in cells.values()):
            none_ok += 1
    for a in ids:
        for b in ids:
            if a < b and by[a].scenario == by[b].scenario:
                same += 1; ab = not main[a][b]["required_conflict"]; ba = not main[b][a]["required_conflict"]
                mutual += int(ab and ba); one_way += int(ab != ba)
    band = Counter(); pub = Counter(); disc = Counter(); missing = Counter()
    for i in ids:
        for j, c in main[i].items():
            if not c.get("different_activity"):
                band["0-3" if c["score"] < 4 else "4-6" if c["score"] < 7 else "7-9"] += 1
                pub[c["status_public"]] += 1; disc[c["status_disclosable"]] += 1
                missing[sum(1 for q in primary[i].requirements if q.required and by[j].facts[q.attribute].visibility != Visibility.public)] += 1
    sw_change = 0
    for rid, sw in m["switches"].items():
        pid = sw["person_id"]
        hi_old = {j for j, c in main[pid].items() if c["score"] >= 7}; hi_new = {j for j, c in sw["row"].items() if c["score"] >= 7}
        col_old = {j for j in sw["column"] if main[j][pid]["score"] >= 7}; col_new = {j for j, c in sw["column"].items() if c["score"] >= 7}
        sw_change += int(hi_old != hi_new or col_old != col_new)
    req_vecs = Counter(json.dumps(sorted((q.attribute, tuple(q.accepted_values), q.required) for q in r.requirements)) for r in requests)
    return {"n_people": len(people), "n_requests": len(requests), "n_switches": len(m["switches"]),
            "scenario_counts": dict(Counter(p.scenario for p in people)),
            "strict_requests": sum(r.strict for r in requests if r.kind == "primary"),
            "required_count_dist": {str(k): v for k, v in sorted(Counter(sum(q.required for q in r.requirements) for r in requests).items())},
            "requirement_sources": dict(Counter(v for r in requests for v in r.sources.values())),
            "switch_factors": dict(Counter(r.change["factor"] for r in requests if r.kind == "alternative")),
            "duplicate_requirement_vectors": sum(v - 1 for v in req_vecs.values()),
            "same_activity_cells": sum(band.values()), "score_band_counts": dict(band),
            "high_partner_count_per_person": {str(k): v for k, v in sorted(high.items())},
            "people_with_no_acceptable_partner": none_ok, "same_activity_pairs": same, "mutual_acceptable_pairs": mutual,
            "one_way_acceptable_pairs": one_way, "switches_changing_high_sets": sw_change,
            "status_public_counts": dict(pub), "status_disclosable_counts": dict(disc),
            "missing_required_facts_per_cell_public": {str(k): v for k, v in sorted(missing.items())},
            "length_band_counts": dict(Counter(p.length_band for p in people)), "text_style_counts": dict(Counter(p.text_style for p in people)),
            "names_unique": len({p.name for p in people}) == len(people)}


# ----------------------------------------------------------------------------- notes for text generation + structured export

def to_persona(p: CommunityPerson, r: RequestSpec) -> Persona:
    st = current_state(p, r)
    facts = {a: Fact(attribute=a, value=st[a], visibility=f.visibility, evidence_id=f.evidence_id) for a, f in p.facts.items()}
    return Persona(persona_id=p.person_id, name=p.name, scenario=p.scenario, style=p.style, facts=facts,
                   background=p.background, requirements=r.requirements)


def _req_words(s: ScenarioSpec, q: Requirement) -> str:
    """Neutral clause about the PARTNER ('you ...'); strength is carried only by the label next to it."""
    t = req_sentence(s, q).replace(" This is a must for me.", "").replace("Ideally ", "").replace(", but that is not a deal-breaker.", "").rstrip(".")
    t = t[0].lower() + t[1:]
    for a, b in (("i need someone who is into ", "you are into "), ("i need you to be the one who ", "you are the one who "),
                 ("i need you to ", "you "), ("i need ", "you have "), ("you need to be available ", "you are available "),
                 ("you should be okay with ", "you are okay with "), ("we need to overlap on ", "we overlap on ")):
        if t.startswith(a):
            t = b + t[len(a):]
            break
    return t


def notes_for(vocab: Vocab, p: CommunityPerson, r: RequestSpec) -> dict[str, Any]:
    """Structured notes the text generator receives: own visible facts, own requirements, 'this time' changes. No labels."""
    rp = render_template(vocab, to_persona(p, r))
    s = vocab.scenarios[p.scenario]
    stable = render_template(vocab, to_persona(p, RequestSpec(request_id=r.request_id, person_id=p.person_id, kind=r.kind, requirements=r.requirements)))
    return {"self_notes": [x["text"] for x in stable.fact_sentences if x["visibility"] != "withheld"],
            "public_notes": [x["text"] for x in stable.fact_sentences if x["visibility"] == "public"],
            "this_time": [f"This time: {s.attributes[a].phrases[v]}. Usually: {s.attributes[a].phrases[p.facts[a].value]}." for a, v in r.overrides.items()],
            "request_notes": [{"strength": "MUST" if q.required else ("STRONG WISH" if r.weights.get(q.id, 1) >= 2 else "WISH"), "text": _req_words(s, q),
                               "alternatives": [s.attributes[q.attribute].phrases[v] for v in q.accepted_values]}
                              for q in r.requirements],
            "template_self": stable.self_description, "template_public": stable.public_text, "template_request": rp.request_text}


def export_structured(vocab: Vocab, comm: dict[str, Any], out: Path) -> dict[str, Any]:
    people = [CommunityPerson.model_validate(x) for x in comm["people"]]
    requests = [RequestSpec.model_validate(x) for x in comm["requests"]]
    for d in ("evaluator", "owner_private", "public_directory", "generation"):
        (out / d).mkdir(parents=True, exist_ok=True)
    (out / "evaluator" / "community_private.json").write_text(json.dumps({k: comm[k] for k in ("vocab_version", "vocab_hash", "seed", "people", "requests")}, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "evaluator" / "matrices.json").write_text(json.dumps(comm["matrices"], indent=1), encoding="utf-8")
    notes = {r.request_id: notes_for(vocab, next(p for p in people if p.person_id == r.person_id), r) for r in requests}
    (out / "generation" / "notes.json").write_text(json.dumps(notes, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(comm["report"], indent=1), encoding="utf-8")
    manifest = {"seed": comm["seed"], "n_people": comm["n_people"], "n_requests": len(requests), "vocab_hash": comm["vocab_hash"],
                "pilot": True, "status": "review_pending", "frozen": False, "human_review": "not_reviewed",
                "text_mode": "none_yet", "labels_from": "program rules only", "scoring_rule": comm["matrices"]["rule"],
                "demographics_note": "age band, gender, nationality, hobby are narrative only and enter no rule"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest
