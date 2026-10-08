"""Core data schemas (plan v0.3, section 8).

Design rules encoded here:
- `unknown` is never encoded as False.
- Conditions carry source spans, direction, required flag and knowledge state.
- Agent inputs never carry oracle labels; oracle labels live in evaluator-only files.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Slot(str, Enum):
    self_ = "self"
    want = "want"
    expect = "expect"
    boundary = "boundary"


class Direction(str, Enum):
    """Whose requirement a condition expresses."""
    about_self = "about_self"      # a fact about the owner
    about_other = "about_other"    # a requirement on the counterpart


class KnowledgeState(str, Enum):
    known = "known"
    unknown = "unknown"


class ConditionStatus(str, Enum):
    satisfied = "satisfied"
    conflict = "conflict"
    unknown = "unknown"


class Visibility(str, Enum):
    public = "public"                 # in the public discovery card
    disclosable = "disclosable"       # owner's agent knows it and may answer on request
    withheld = "withheld"             # not given to the owner's agent / must stay unknown


class Decision(str, Enum):
    recommend = "recommend"
    reject = "reject"
    insufficient_info = "insufficient_info"


class ExecutionStatus(str, Enum):
    ok = "ok"
    execution_error = "execution_error"  # timeouts, format failures, budget exhaustion


class MessageType(str, Enum):
    ASK = "ASK"
    ANSWER = "ANSWER"
    STOP = "STOP"


class Answer(str, Enum):
    yes = "yes"
    no = "no"
    unknown = "unknown"


class Condition(BaseModel):
    id: str
    text: str
    source_span: str = Field(description="Verbatim text the condition was extracted from")
    slot: Slot
    direction: Direction
    required: bool = False
    knowledge_state: KnowledgeState = KnowledgeState.known
    vocab_key: Optional[str] = Field(
        default=None,
        description="Key in the scenario condition vocabulary; None means out of vocabulary (kept verbatim, treated as unsupported)",
    )
    value: Optional[str] = None

    confirmed: bool = Field(default=False, description="Owner explicitly confirmed this condition (from the current request or a confirmation step)")

    @property
    def enforceable(self) -> bool:
        """A required condition may drive an automatic reject only if the owner confirmed it."""
        return self.required and self.confirmed


class Evidence(BaseModel):
    id: str
    text: str
    visibility: Visibility


class Profile(BaseModel):
    """Owner-side profile as the owner's agent sees it (natural language first)."""
    persona_id: str
    self_description: str
    evidence: list[Evidence] = Field(default_factory=list)
    scenario: str


class IntentCard(BaseModel):
    persona_id: str
    request_id: str
    request_text: str
    conditions: list[Condition] = Field(default_factory=list)
    extraction_mode: Literal["llm", "gold_card_dev_only"] = "llm"


class DisclosurePolicy(BaseModel):
    persona_id: str
    public_evidence_ids: list[str] = Field(default_factory=list)
    disclosable_evidence_ids: list[str] = Field(default_factory=list)

    def may_send(self, evidence_id: str) -> bool:
        return evidence_id in self.public_evidence_ids or evidence_id in self.disclosable_evidence_ids


class PublicCard(BaseModel):
    persona_id: str
    activity: str
    time_window: Optional[str] = None
    region: Optional[str] = None
    public_text: str


class Message(BaseModel):
    session_id: str
    turn: int
    sender: str
    recipient: str
    type: MessageType
    condition_id: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[Answer] = None
    evidence_ids: list[str] = Field(default_factory=list)
    note: Optional[str] = None


class ConditionAssessment(BaseModel):
    condition_id: str
    status: ConditionStatus
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = ""
    evidence_quote: str = ""
    llm_status: Optional[ConditionStatus] = None     # what the model said before any program guard
    guard: Optional[str] = None                      # e.g. no_verbatim_evidence, answered_yes, answered_no, answered_unknown


class Assessment(BaseModel):
    """Directional assessment: `assessor` judging `target` against assessor's own conditions."""
    assessor: str
    target: str
    conditions: list[ConditionAssessment]
    decision: Decision                                   # program decision
    unresolved: list[str] = Field(default_factory=list)
    proposed: Optional[Decision] = None                  # LLM proposal, comparison only
    proposal_matches: Optional[bool] = None


class Usage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    calls: int = 0
    retries: int = 0
    latency_s: float = 0.0
    tokens_missing: bool = False  # backend did not report usage; do not fill with zero


class MatchResult(BaseModel):
    pair_id: str
    method: str
    decision_ab: Optional[Decision] = None          # program-aggregated from condition states
    decision_ba: Optional[Decision] = None
    proposed_ab: Optional[Decision] = None          # what the LLM proposed, kept for comparison only
    proposed_ba: Optional[Decision] = None
    proposal_matches_ab: Optional[bool] = None
    proposal_matches_ba: Optional[bool] = None
    joint_decision: Optional[Decision] = None
    unresolved: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    execution_status: ExecutionStatus = ExecutionStatus.ok
    error: Optional[str] = None
    detail: dict = Field(default_factory=dict)


class RequiredStateError(ValueError):
    pass


def _statuses(assessments: list["ConditionAssessment"], conditions: list[Condition]) -> dict[str, ConditionStatus]:
    ids = [a.condition_id for a in assessments]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise RequiredStateError(f"duplicate condition ids in assessment: {dup}")
    known = {c.id for c in conditions}
    unknown_ids = sorted(set(ids) - known)
    if unknown_ids:
        raise RequiredStateError(f"assessment refers to unknown condition ids: {unknown_ids}")
    return {a.condition_id: a.status for a in assessments}


def decide_direction(assessments: list["ConditionAssessment"], conditions: list[Condition]) -> Decision:
    """Program-side directional decision from condition statuses .

    Only required AND confirmed conditions are enforceable. conflict -> reject; unknown -> insufficient_info;
    all enforceable satisfied -> recommend. Preferences never change the class. This is the final decision;
    an LLM-proposed decision is stored separately and compared (see reconcile_decision).
    """
    statuses = _statuses(assessments, conditions)
    enforceable = [c for c in conditions if c.enforceable]
    if any(statuses.get(c.id) == ConditionStatus.conflict for c in enforceable):
        return Decision.reject
    if any(statuses.get(c.id, ConditionStatus.unknown) == ConditionStatus.unknown for c in enforceable):
        return Decision.insufficient_info
    return Decision.recommend


def reconcile_decision(proposed: Optional[Decision], assessments: list["ConditionAssessment"],
                       conditions: list[Condition]) -> tuple[Decision, bool]:
    """Final decision always comes from the program. Returns (final, proposal_matches)."""
    final = decide_direction(assessments, conditions)
    return final, (proposed is None or proposed == final)


def validate_decision(decision: Decision, assessments: list["ConditionAssessment"], conditions: list[Condition]) -> None:
    """Raise if a proposed decision differs from the program decision in ANY direction (not only wrong recommends)."""
    final, ok = reconcile_decision(decision, assessments, conditions)
    if not ok:
        raise RequiredStateError(f"proposed {decision.value} but condition states imply {final.value}")


def aggregate_joint(a: Decision, b: Decision) -> Decision:
    """Program-side aggregation . A single reject cannot be overridden."""
    if Decision.reject in (a, b):
        return Decision.reject
    if a == Decision.recommend and b == Decision.recommend:
        return Decision.recommend
    return Decision.insufficient_info
