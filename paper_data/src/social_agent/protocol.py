"""Two agents, one pair: no conversation (B2) or a shared budget of yes/no questions (B3ft family).
Answers are computed by the program from facts the answering agent extracted from its owner's own texts; after an answer,
the program updates only the asked condition. The decision of each side and the handshake follow schemas.decide_direction /
aggregate_joint. The protocol never reads the reference answers."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from social_agent.agent import AgentError, PersonalAgent, build_agent_context
from social_agent.vocab import Vocab
from social_agent.schemas import Decision, ExecutionStatus, MatchResult, Message, MessageType, Usage, aggregate_joint


def _sum_usage(agents: list[PersonalAgent], wall: float) -> Usage:
    rows = [u for a in agents for u in a.usage]
    inc = any(u["tokens_incomplete"] for u in rows)
    return Usage(prompt_tokens=None if inc else sum(u["prompt_tokens"] or 0 for u in rows),
                 completion_tokens=None if inc else sum(u["completion_tokens"] or 0 for u in rows),
                 calls=sum(u["calls"] for u in rows), retries=sum(u["network_retries"] + u["json_repairs"] for u in rows),
                 latency_s=wall, tokens_missing=inc)


async def run_pair(agent_inputs: dict[str, Any], vocab: Vocab, pair: dict[str, str], client, *, method: str,
                   max_questions: int, cache_dir: Optional[Path], cache_key_parts: tuple) -> MatchResult:
    t0 = time.perf_counter()
    a_id, b_id, pid = pair["requester_id"], pair["candidate_id"], pair["pair_id"]
    scen = vocab.scenarios[agent_inputs["personas"][a_id]["scenario"]]
    A = PersonalAgent(build_agent_context(agent_inputs, a_id, b_id), scen, client, cache_dir=cache_dir, cache_key_parts=cache_key_parts)
    B = PersonalAgent(build_agent_context(agent_inputs, b_id, a_id), scen, client, cache_dir=cache_dir, cache_key_parts=cache_key_parts)
    res = MatchResult(pair_id=pid, method=method)
    log: list[dict[str, Any]] = []
    try:
        await A.extract(); await B.extract()
        if max_questions > 0:
            await A.extract_self_facts_from_text(); await B.extract_self_facts_from_text()
        log.append({"state": "assessing"})
        ab, ba = await A.assess(), await B.assess()
        asked: set[str] = set()
        turn = 0
        session = f"{pid}:{method}"
        # clarifying: shared budget, alternate, program picks the condition, stop on known reject
        while max_questions > 0 and turn < max_questions:
            if Decision.reject in (ab.decision, ba.decision):
                break
            order = [(A, B), (B, A)] if turn % 2 == 0 else [(B, A), (A, B)]
            picked = None
            for asker, holder in order:
                c = asker.next_unresolved(asked)
                if c:
                    picked = (asker, holder, c); break
            if not picked:
                break
            asker, holder, cond = picked
            asked.add(cond.id)
            turn += 1
            q = await asker.ask(cond, session, turn)
            holder.ctx.received_messages.append(q)          # ASK is the only thing the holder receives
            ans = holder.answer_from_facts(q, turn)
            asker.receive_answer(ans)
            res.messages += [q, ans]
            log.append({"state": "clarifying", "turn": turn, "asker": asker.pid, "condition": cond.id, "answer": ans.answer.value})
            # program-only update of the answered condition; no new LLM judgement
            ab, ba = (asker.update_from_answers() if asker is A else ab), (asker.update_from_answers() if asker is B else ba)
        res.decision_ab, res.decision_ba = ab.decision, ba.decision
        res.proposed_ab, res.proposed_ba = ab.proposed, ba.proposed
        res.proposal_matches_ab, res.proposal_matches_ba = ab.proposal_matches, ba.proposal_matches
        res.joint_decision = aggregate_joint(ab.decision, ba.decision)
        res.unresolved = ab.unresolved + ba.unresolved
        res.evidence = [e for m in res.messages for e in m.evidence_ids]
        res.execution_status = ExecutionStatus.ok
        res.detail = {"conditions_a": [c.model_dump() for c in A.conditions], "conditions_b": [c.model_dump() for c in B.conditions],
                      "assess_a": ab.model_dump(), "assess_b": ba.model_dump(), "questions_used": turn, "log": log,
                      "extraction_a": A.extraction_raw, "extraction_b": B.extraction_raw,
                      "self_facts_a": getattr(A, "self_facts", None), "self_facts_b": getattr(B, "self_facts", None),
                      "usage_steps": A.usage + B.usage}
    except AgentError as e:
        res.execution_status, res.error = ExecutionStatus.execution_error, str(e)
        res.detail = {"log": log, "usage_steps": A.usage + B.usage}
    res.usage = _sum_usage([A, B], time.perf_counter() - t0)
    return res


# Methods reported in the paper. B2 = no conversation; B3ft* = up to N questions, answers computed from the owner's texts.
METHODS = {"B2": {"max_questions": 0, "desc": "two agents, no conversation"},
           "B3ft_q1": {"max_questions": 1, "desc": "two agents, at most 1 question per pair"},
           "B3ft": {"max_questions": 3, "desc": "two agents, at most 3 questions per pair"},
           "B3ft_q5": {"max_questions": 5, "desc": "two agents, at most 5 questions per pair"}}
