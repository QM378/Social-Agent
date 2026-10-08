import pytest
from social_agent.schemas import (Decision, aggregate_joint, ConditionStatus, Condition, ConditionAssessment, Slot,
                                  Direction, KnowledgeState, decide_direction, validate_decision, RequiredStateError)


def _c(cid, required=True, confirmed=True):
    return Condition(id=cid, text=cid, source_span=cid, slot=Slot.expect, direction=Direction.about_other,
                     required=required, confirmed=confirmed)


def _a(cid, st):
    return ConditionAssessment(condition_id=cid, status=st)


def test_single_reject_cannot_be_overridden():
    assert aggregate_joint(Decision.recommend, Decision.reject) == Decision.reject
    assert aggregate_joint(Decision.reject, Decision.recommend) == Decision.reject


def test_unknown_blocks_recommend():
    assert aggregate_joint(Decision.recommend, Decision.insufficient_info) == Decision.insufficient_info


def test_required_conflict_rejects_required_unknown_insufficient():
    cs = [_c("c1"), _c("c2")]
    assert decide_direction([_a("c1", ConditionStatus.conflict), _a("c2", ConditionStatus.satisfied)], cs) == Decision.reject
    assert decide_direction([_a("c1", ConditionStatus.unknown), _a("c2", ConditionStatus.satisfied)], cs) == Decision.insufficient_info
    assert decide_direction([_a("c1", ConditionStatus.satisfied), _a("c2", ConditionStatus.satisfied)], cs) == Decision.recommend


def test_unconfirmed_boundary_is_not_enforceable():
    cs = [_c("c1"), _c("b1", required=True, confirmed=False)]
    assert decide_direction([_a("c1", ConditionStatus.satisfied), _a("b1", ConditionStatus.conflict)], cs) == Decision.recommend
    assert not cs[1].enforceable


def test_preference_conflict_does_not_reject():
    cs = [_c("c1"), _c("p1", required=False)]
    assert decide_direction([_a("c1", ConditionStatus.satisfied), _a("p1", ConditionStatus.conflict)], cs) == Decision.recommend


def test_validate_decision_rejects_recommend_with_unknown_required():
    cs = [_c("c1")]
    with pytest.raises(RequiredStateError):
        validate_decision(Decision.recommend, [_a("c1", ConditionStatus.unknown)], cs)
    validate_decision(Decision.recommend, [_a("c1", ConditionStatus.satisfied)], cs)


def test_validate_decision_catches_insufficient_when_conflict_known():
    cs = [_c("c1")]
    with pytest.raises(RequiredStateError):
        validate_decision(Decision.insufficient_info, [_a("c1", ConditionStatus.conflict)], cs)
    with pytest.raises(RequiredStateError):
        validate_decision(Decision.reject, [_a("c1", ConditionStatus.satisfied)], cs)


def test_duplicate_and_unknown_condition_ids_rejected():
    cs = [_c("c1")]
    with pytest.raises(RequiredStateError):
        decide_direction([_a("c1", ConditionStatus.satisfied), _a("c1", ConditionStatus.conflict)], cs)
    with pytest.raises(RequiredStateError):
        decide_direction([_a("zz", ConditionStatus.satisfied)], cs)


def test_reconcile_keeps_program_decision():
    from social_agent.schemas import reconcile_decision
    cs = [_c("c1")]
    final, ok = reconcile_decision(Decision.recommend, [_a("c1", ConditionStatus.unknown)], cs)
    assert final == Decision.insufficient_info and not ok
