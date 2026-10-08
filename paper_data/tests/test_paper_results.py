import json

from social_agent.evaluation.score import _raw, audit, filter_runs, planned_pairs


def test_common_set_and_full_cost(tmp_path):
    """Two methods, different completion: scoring uses only pairs both completed; cost counts every record."""
    pools = {"pools": {"u1.r1": {"person_id": "u1", "pool": ["u2", "u3"]}}}
    for m, rows in {"A": [("u2", "ok", 1), ("u3", "ok", 1)], "B": [("u2", "ok", 2), ("u3", "execution_error", 5)]}.items():
        d = tmp_path / m; d.mkdir()
        (d / "pools.json").write_text(json.dumps(pools))
        (d / "cases.jsonl").write_text("\n".join(json.dumps({"request_id": "u1.r1", "person_id": "u1", "candidate_id": j, "method": m,
                                                             "request_kind": "primary", "execution_status": st, "decision_ab": "recommend",
                                                             "usage": {"calls": c}}) for j, st, c in rows))
    plan = planned_pairs(tmp_path / "A" / "pools.json")
    rep, common = audit({"A": tmp_path / "A", "B": tmp_path / "B"}, plan)
    assert plan == {("u1.r1", "u2"), ("u1.r1", "u3")} and common == {("u1.r1", "u2")}
    assert rep["B"]["execution_error"] == 1 and rep["A"]["completed_ok"] == 2
    _, cost_b = _raw(tmp_path / "B", "B")
    assert cost_b["calls"] == 7
    f = filter_runs({"B": tmp_path / "B"}, common, tmp_path / "filtered")
    assert len((f["B"] / "cases.jsonl").read_text().splitlines()) == 1
