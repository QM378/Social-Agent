"""The attribute vocabulary (configs/vocab.yaml) and the structured types of the simulated community: facts, their
visibility, requirements, and how a stated value is checked against a requirement.

Nothing in this module calls an LLM.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field

from social_agent.schemas import ConditionStatus, Decision, Visibility, aggregate_joint

# ----------------------------------------------------------------------------- vocab

class AttrSpec(BaseModel):
    name: str
    kind: Literal["enum", "set"]
    values: list[str]
    default_visibility: Visibility
    accepts: Optional[dict[str, Optional[list[str]]]] = None
    phrases: dict[str, str]
    meaning: dict[str, str] = Field(default_factory=dict)   # value glossary as a property of the person holding it


class ScenarioSpec(BaseModel):
    label: str
    attributes: dict[str, AttrSpec]
    background_attrs: list[str] = Field(default_factory=list)
    background_values: dict[str, list[str]] = Field(default_factory=dict)


class Vocab(BaseModel):
    version: str
    scenarios: dict[str, ScenarioSpec]
    source_hash: str = ""


def load_vocab(path: str | Path | None = None) -> Vocab:
    from social_agent.paths import VOCAB
    path = VOCAB if path is None else path
    raw = Path(path).read_text(encoding="utf-8")
    d = yaml.safe_load(raw)
    scen = {}
    for sname, s in d["scenarios"].items():
        attrs = {a: AttrSpec(name=a, **spec) for a, spec in s["attributes"].items()}
        scen[sname] = ScenarioSpec(label=s["label"], attributes=attrs,
                                   background_attrs=s.get("background_attrs", []),
                                   background_values=s.get("background_values", {}))
    return Vocab(version=d["version"], scenarios=scen, source_hash=hashlib.sha256(raw.encode()).hexdigest()[:16])


# ----------------------------------------------------------------------------- world objects

class Requirement(BaseModel):
    id: str
    attribute: str
    operator: Literal["in", "intersects"]
    accepted_values: list[str]
    required: bool
    confirmed: bool                 # required conditions come from the explicit request and are confirmed
    source: Literal["request", "preference"]


class Fact(BaseModel):
    attribute: str
    value: Optional[str | list[str]] = None   # None = truth itself undefined (unknown)
    visibility: Visibility
    evidence_id: str


class Persona(BaseModel):
    persona_id: str
    name: str
    scenario: str
    style: str
    facts: dict[str, Fact]
    background: dict[str, str] = Field(default_factory=dict)
    requirements: list[Requirement] = Field(default_factory=list)


ROLE_ATTRS = {"photography": "editing_role", "gaming": "coaching_role", "study": "tutor_role"}
_NAMES = [
    "Avery", "Blake", "Casey", "Dana", "Ellis", "Finley", "Gray", "Harper", "Indigo", "Jules", "Kai", "Lane",
    "Morgan", "Noor", "Oakley", "Parker", "Quinn", "Reese", "Sage", "Tatum", "Uma", "Val", "Wren", "Xen", "Yael",
    "Zion", "Adair", "Bex", "Cruz", "Devin", "Emery", "Frankie", "Greer", "Hollis", "Ira", "Jordan", "Kit", "Lior",
    "Marlow", "Nico", "Onyx", "Peyton", "Rio", "Sky", "Toby", "Ursa", "Vesper", "Winter", "Ash", "Bay", "Cy", "Dell",
    "Eden", "Fern", "Gale", "Haven", "Isa", "Jem", "Kes", "Lux", "Mica", "Nell", "Ode", "Pax", "Rae", "Sol", "Tam",
    "Vik", "Wes", "Zed", "Arden", "Brook", "Cove", "Dale", "Echo", "Flynn", "Glen", "Hale", "Ines", "Jay", "Koa",
    "Lark", "Moss", "Nova", "Orla", "Pip", "Rowan", "Shay", "Teal", "Vale", "Wynn", "Ace", "Bo", "Cal", "Dov", "Eli",
    "Fox", "Gus", "Hux", "Ivy", "Jo", "Kip", "Lev", "Max", "Ned", "Oz", "Pat", "Ren", "Sam", "Tal", "Uri", "Vin",
    "Wil", "Yan", "Zev", "Alba", "Bram", "Cleo", "Dara", "Esme", "Fitz", "Gwen", "Hana", "Ivo", "Juno", "Kira",
]
STYLES = ["plain", "chatty", "terse"]


# ----------------------------------------------------------------------------- evaluation rules

def eval_requirement(req: Requirement, fact_value: Optional[str | list[str]]) -> ConditionStatus:
    """Unknown fact -> unknown. Enum: membership. Set: non-empty intersection."""
    if fact_value is None:
        return ConditionStatus.unknown
    if req.operator == "in":
        return ConditionStatus.satisfied if fact_value in req.accepted_values else ConditionStatus.conflict
    vals = set(fact_value) if isinstance(fact_value, list) else {fact_value}
    return ConditionStatus.satisfied if vals & set(req.accepted_values) else ConditionStatus.conflict
