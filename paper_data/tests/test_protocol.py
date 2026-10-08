"""The method on the real community, with a mock model: what agents may see, the question budget, program answers,
the decision rule, and the method table. No Ollama needed."""
import asyncio
import json

from social_agent.agent import build_agent_context
from social_agent.experiment.runner import load_community_inputs
from social_agent.llm import LLMConfig, MockClient
from social_agent.mock import mock_responder
from social_agent.paths import DATA
from social_agent.protocol import METHODS, run_pair
from social_agent.schemas import Decision, aggregate_joint
from social_agent.vocab import load_vocab

INPUTS = load_community_inputs(DATA)
V = load_vocab()


def _pair(i=0):
    rid = next(r["request_id"] for r in INPUTS["requests"] if r["kind"] == "primary")
    a = rid.split(".")[0]
    b = next(p for p, x in INPUTS["personas"].items() if p != a and x["scenario"] == INPUTS["personas"][a]["scenario"])
    return {"pair_id": f"{rid}|{b}", "requester_id": a, "candidate_id": b}


def test_methods_are_exactly_the_reported_ones():
    assert {m: v["max_questions"] for m, v in METHODS.items()} == {"B2": 0, "B3ft_q1": 1, "B3ft": 3, "B3ft_q5": 5}


def test_agents_never_see_reference_answers_or_the_other_persons_private_text():
    p = _pair()
    ctx = build_agent_context(INPUTS, p["requester_id"], p["candidate_id"])
    other = INPUTS["personas"][p["candidate_id"]]
    blob = json.dumps(ctx.model_dump(mode="json"))
    assert other["self_description"] not in blob and other["request_text"] not in blob
    assert set(ctx.other_public_card) <= {"persona_id", "name", "scenario", "public_text"}
    assert "matrices" not in blob and "status_disclosable" not in blob


def test_question_budget_is_respected_and_answers_come_from_the_program():
    p = _pair()
    for m, spec in METHODS.items():
        r = asyncio.run(run_pair(INPUTS, V, p, MockClient(LLMConfig(), responder=mock_responder), method=m,
                                 max_questions=spec["max_questions"], cache_dir=None, cache_key_parts=("t",)))
        asks = [x for x in r.messages if x.type.value == "ASK"]
        answers = [x for x in r.messages if x.type.value == "ANSWER"]
        assert len(asks) <= spec["max_questions"] and len(asks) == len(answers)
        assert all(a.answer.value in ("yes", "no", "unknown") for a in answers)
        assert r.joint_decision == aggregate_joint(r.decision_ab, r.decision_ba)


def test_handshake_needs_both_sides():
    assert aggregate_joint(Decision.recommend, Decision.recommend) == Decision.recommend
    assert aggregate_joint(Decision.recommend, Decision.reject) == Decision.reject
    assert aggregate_joint(Decision.recommend, Decision.insufficient_info) == Decision.insufficient_info
