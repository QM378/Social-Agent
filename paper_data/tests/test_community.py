import asyncio, json
from pathlib import Path
import pytest

from social_agent.community.generate import (generate_community, export_structured, compute_matrices, check_consistency,
                                          CommunityPerson, RequestSpec, current_state, FEASIBILITY_ATTRS)
from social_agent.community import texts as ct
from social_agent.vocab import load_vocab
from social_agent.llm import LLMConfig, MockClient
from social_agent.schemas import Visibility

V = load_vocab()


def test_generation_rules_and_determinism():
    a = generate_community(V, seed=3, n_people=30); b = generate_community(V, seed=3, n_people=30)
    assert a["people"] == b["people"] and a["matrices"]["main"] == b["matrices"]["main"]
    people = [CommunityPerson.model_validate(p) for p in a["people"]]; reqs = [RequestSpec.model_validate(r) for r in a["requests"]]
    check_consistency(V, people, reqs)                      # withheld/capability/feasibility/preference rules
    by = {p.person_id: p for p in people}
    for r in reqs:
        p = by[r.person_id]
        assert not any(p.facts[q.attribute].visibility == Visibility.withheld for q in r.requirements)
        assert any(q.required for q in r.requirements) and any(not q.required for q in r.requirements)
        assert all(a not in FEASIBILITY_ATTRS for a in r.overrides)
        if r.kind == "alternative":
            assert r.change and r.change["factor"] in ("activity", "relax", "willingness")
    assert 1 <= sum(r.kind == "alternative" for r in reqs) <= 8


def test_matrices_directional_and_switch_row_and_column():
    a = generate_community(V, seed=4, n_people=24)
    m = a["matrices"]; ids = m["person_ids"]
    for i in ids:
        assert i not in m["main"][i] and len(m["main"][i]) == len(ids) - 1
        for j, c in m["main"][i].items():
            assert 0 <= c["score"] <= 9 and ((c["score"] < 4) == c["required_conflict"])
    assert any(m["main"][i][j]["score"] != m["main"][j][i]["score"] for i in ids for j in ids if i != j)
    for rid, sw in m["switches"].items():
        pid = sw["person_id"]
        assert set(sw["row"]) == set(ids) - {pid}
        assert all(j != pid for j in sw["column"])
        # the switch only changes this person: other people's rows in the switched state only differ in column pid
        # (column values come from others' primary requests evaluated against the new state)
        assert isinstance(sw["change"], dict)


def test_demographics_do_not_affect_matrices():
    a = generate_community(V, seed=5, n_people=15)
    people = [CommunityPerson.model_validate(p) for p in a["people"]]; reqs = [RequestSpec.model_validate(r) for r in a["requests"]]
    for p in people:
        p.gender, p.nationality, p.age_band, p.hobby, p.name = "X", "Y", "Z", "W", "N"
    assert compute_matrices(V, people, reqs)["main"] == a["matrices"]["main"]


def test_notes_contain_no_labels_and_text_pipeline_with_mock(tmp_path, monkeypatch):
    a = generate_community(V, seed=6, n_people=9)
    out = tmp_path / "c"
    export_structured(V, a, out)
    notes = json.loads((out / "generation" / "notes.json").read_text())
    blob = json.dumps(notes)
    for k in ("score", "required_conflict", "status_public", "accepted_values", "u\"", "withheld"):
        assert k not in blob
    assignment = ct.assign_models(a, ["gemma", "gptoss"], 1)
    (out / "generation" / "assignment.json").write_text(json.dumps(assignment))
    assert set(assignment.values()) == {"gemma", "gptoss"}

    def responder(system, user, schema):
        if schema.__name__ == "_Text":
            base = user.split("Facts (stable):", 1)[-1] if "Facts (stable)" in user else user.split("Requirements:", 1)[-1]
            words = " ".join(w for w in base.replace("-", " ").split() if not w.isupper())
            return {"text": ("I am a fictional person. " + words + " ") * 3}
        if schema.__name__ == "_ExtractedFacts":
            return {"facts": {}}
        return {"requirements": []}
    monkeypatch.setattr(ct, "make_client", lambda llm, budget=None: MockClient(llm, responder=responder, budget=budget))
    r1 = asyncio.run(ct.render_community(out, "gemma", LLMConfig()))
    r2 = asyncio.run(ct.render_community(out, "gptoss", LLMConfig()))
    assert r1["planned"] + r2["planned"] == 9 and r1["skipped_done"] == 0
    r3 = asyncio.run(ct.render_community(out, "gemma", LLMConfig()))       # resume
    assert r3["planned"] == r1["planned"] and r3["done"] == 0 and r3["skipped_done"] + r3["needs_review"] + r3["failed"] == r1["planned"]
    c = asyncio.run(ct.crosscheck_community(out, "gptoss", LLMConfig()))
    assert c["checked"] == r1["planned"]
    m = ct.assemble_community(out)
    pub = (out / "public_directory" / "directory.jsonl").read_text()
    assert "self_text" not in pub and "request_text" not in pub and "score" not in pub
    priv = (out / "owner_private" / "u0001.json").read_text()
    assert "score" not in priv and "required_conflict" not in priv
    assert not (out / "full_info_view").exists() and not (out / "owner_private_debug").exists()


def test_alternative_changes_only_one_factor():
    a = generate_community(V, seed=0, n_people=120)
    prim = {r["person_id"]: r for r in a["requests"] if r["kind"] == "primary"}
    key = lambda q: (q["attribute"], tuple(q["accepted_values"]), q["required"])
    n = 0
    for r in a["requests"]:
        if r["kind"] != "alternative":
            continue
        n += 1
        p = prim[r["person_id"]]; ch = r["change"]
        A = {key(q): q for q in p["requirements"]}; B = {key(q): q for q in r["requirements"]}
        changed = {q["attribute"] for k, q in A.items() if k not in B} | {q["attribute"] for k, q in B.items() if k not in A}
        allowed = {ch["attribute"]} | ({"editing_role", "coaching_role", "tutor_role"} if ch["factor"] != "relax" else set())
        assert changed <= allowed, (r["request_id"], ch, changed)
        for k, q in B.items():
            if k in A and not q["required"]:
                assert r["weights"].get(q["id"]) == p["weights"].get(A[k]["id"], 1.0) or ch["factor"] == "relax"
    assert n >= 10


def test_resume_gate_and_crosscheck_states(tmp_path, monkeypatch):
    a = generate_community(V, seed=8, n_people=6)
    out = tmp_path / "c"; export_structured(V, a, out)
    (out / "generation" / "assignment.json").write_text(json.dumps({p["person_id"]: "gemma" for p in a["people"]}))
    calls = {"n": 0}

    def responder(system, user, schema):
        calls["n"] += 1
        if schema.__name__ == "_Text":
            base = user.split("Facts (stable):", 1)[-1] if "Facts (stable)" in user else user.split("Requirements:", 1)[-1]
            return {"text": ("Fictional person. " + " ".join(w for w in base.replace("-", " ").split() if not w.isupper()) + " ") * 3}
        return {"facts": {}} if schema.__name__ == "_ExtractedFacts" else {"requirements": []}
    monkeypatch.setattr(ct, "make_client", lambda llm, budget=None: MockClient(llm, responder=responder, budget=budget))
    r1 = asyncio.run(ct.render_community(out, "gemma", LLMConfig(), limit=3))
    st = ct.model_status(out, "gemma")
    assert st["smoke_pass"] and st["usable"] == 3 and st["missing"] == 3
    r2 = asyncio.run(ct.render_community(out, "gemma", LLMConfig()))          # continues the rest, skips the complete ones
    assert r2["skipped_done"] + r2["done"] + r2["needs_review"] == 6 and r2["usable"] == 6
    complete_before = ct.model_status(out, "gemma")["complete"]
    # a person with a missing request text is NOT complete
    pid = a["people"][0]["person_id"]; f = out / "generation" / "texts" / f"{pid}.json"
    rec = json.loads(f.read_text()); rec["requests"] = {}; f.write_text(json.dumps(rec))
    assert not ct.text_record_complete(rec, ct.expected_request_ids(json.loads((out / "evaluator" / "community_private.json").read_text()), pid))
    before = calls["n"]
    r3 = asyncio.run(ct.render_community(out, "gemma", LLMConfig()))
    assert r3["skipped_done"] == complete_before - 1 and calls["n"] > before   # regenerated (needs_review ones retried too), history kept
    assert json.loads(f.read_text())["history"]
    # cross-check states: complete -> clean; then change a text -> re-checked (hash bound); incomplete -> "incomplete"
    c1 = asyncio.run(ct.crosscheck_community(out, "gptoss", LLMConfig()))
    assert c1["checked"] == 6 and set(c1["verdicts"]) <= {"clean", "needs_adjudication", "wording_only", "incomplete"}
    rec = json.loads(f.read_text()); rec["self"]["text"] += " Changed."; f.write_text(json.dumps(rec))
    c2 = asyncio.run(ct.crosscheck_community(out, "gptoss", LLMConfig()))
    assert c2["checked"] == 1
    rec = json.loads(f.read_text()); rec["self"]["status"] = "failed"; rec["self"]["text"] = ""; f.write_text(json.dumps(rec))
    c3 = asyncio.run(ct.crosscheck_community(out, "gptoss", LLMConfig()))
    assert c3["verdicts"] == {"incomplete": 1}
    sm = ct.summary(out)
    assert sm["all_complete"] is False and sm["people_text_status"]["failed"] == 1


def test_provenance_and_status_fields_present(tmp_path, monkeypatch):
    a = generate_community(V, seed=9, n_people=4)
    out = tmp_path / "c"; export_structured(V, a, out)
    (out / "generation" / "assignment.json").write_text(json.dumps({p["person_id"]: "gemma" for p in a["people"]}))

    def responder(system, user, schema):
        if schema.__name__ == "_Text":
            base = user.split("Facts (stable):", 1)[-1] if "Facts (stable)" in user else user.split("Requirements:", 1)[-1]
            return {"text": ("Fictional person. " + " ".join(w for w in base.replace("-", " ").split() if not w.isupper()) + " ") * 3}
        return {"facts": {}} if schema.__name__ == "_ExtractedFacts" else {"requirements": [], "changes_this_time": []}
    monkeypatch.setattr(ct, "make_client", lambda llm, budget=None: MockClient(llm, responder=responder, budget=budget))
    asyncio.run(ct.render_community(out, "gemma", LLMConfig()))
    rec = json.loads(next((out / "generation" / "texts").glob("u*.json")).read_text())
    att = rec["self"]["attempts"][0]
    assert "prompt_sent" in att and att["prompt_sent"]["system"] and att["prompt_sent"]["user"] and "raw_output" in att and "seed" in att
    for k in ("model_digest", "model_quantization", "rules_version", "code_version", "generation_status", "auto_check_status", "human_review_status"):
        assert k in rec
    assert rec["auto_check_status"] == "not_checked" and rec["human_review_status"] == "not_reviewed"
    prov = json.loads((out / "generation" / "provenance_gemma.json").read_text())
    assert prov["digest"] == "unknown" and prov["prompt_version"] and prov["code_version"]      # unknown, never guessed
    asyncio.run(ct.crosscheck_community(out, "gptoss", LLMConfig()))
    rec = json.loads(next((out / "generation" / "texts").glob("u*.json")).read_text())
    assert rec["auto_check_status"] in ("clean", "needs_adjudication", "wording_only", "incomplete", "checker_error")
    m = ct.assemble_community(out)
    assert m["pilot"] is True and m["frozen"] is False and m["status"] == "review_pending"
    priv = json.loads((out / "owner_private" / "u0001.json").read_text())
    assert {"generation_status", "auto_check_status", "human_review_status", "provenance"} <= set(priv)


def test_crosscheck_covers_weights_and_this_time(tmp_path, monkeypatch):
    a = generate_community(V, seed=0, n_people=40)
    alt = next(r for r in a["requests"] if r["kind"] == "alternative" and r["overrides"])
    pid = alt["person_id"]
    out = tmp_path / "c"; export_structured(V, a, out)
    (out / "generation" / "assignment.json").write_text(json.dumps({pid: "gemma"}))
    prim = next(r for r in a["requests"] if r["person_id"] == pid and r["kind"] == "primary")

    def responder(system, user, schema):
        if schema.__name__ == "_Text":
            return {"text": "Fictional text about a person that is long enough to pass the structural checks for length and content here. " * 2}
        if schema.__name__ == "_ExtractedFacts":
            return {"facts": {}}
        # gold-perfect requirements but wrong weights and no this-time change
        gold = alt if "alt" in user else prim
        return {"requirements": [{"attribute": q["attribute"], "accepted_values": q["accepted_values"],
                                  "strength": "required" if q["required"] else "preference"} for q in alt["requirements"]],
                "changes_this_time": []}
    monkeypatch.setattr(ct, "make_client", lambda llm, budget=None: MockClient(llm, responder=responder, budget=budget))
    asyncio.run(ct.render_community(out, "gemma", LLMConfig()))
    asyncio.run(ct.crosscheck_community(out, "gptoss", LLMConfig()))
    chk = json.loads((out / "generation" / "crosscheck" / f"{pid}.json").read_text())
    types = {i["type"] for i in chk["issues"] if i["where"] == alt["request_id"]}
    assert "this_time_missing" in types
    if any(alt["weights"].get(q["id"], 1.0) >= 2 for q in alt["requirements"] if not q["required"]):
        assert "weight_changed" in types


def test_http_attempt_level_provenance_and_stale_check_is_not_checked(tmp_path, monkeypatch):
    a = generate_community(V, seed=10, n_people=3)
    out = tmp_path / "c"; export_structured(V, a, out)
    (out / "generation" / "assignment.json").write_text(json.dumps({p["person_id"]: "gemma" for p in a["people"]}))

    def responder(system, user, schema):
        if schema.__name__ == "_Text":
            base = user.split("Facts (stable):", 1)[-1] if "Facts (stable)" in user else user.split("Requirements:", 1)[-1]
            return {"text": ("Fictional person. " + " ".join(w for w in base.replace("-", " ").split() if not w.isupper()) + " ") * 3}
        return {"facts": {}} if schema.__name__ == "_ExtractedFacts" else {"requirements": [], "changes_this_time": []}
    monkeypatch.setattr(ct, "make_client", lambda llm, budget=None: MockClient(llm, responder=responder, budget=budget))
    asyncio.run(ct.render_community(out, "gemma", LLMConfig()))
    asyncio.run(ct.crosscheck_community(out, "gptoss", LLMConfig()))
    f = next((out / "generation" / "texts").glob("u*.json")); rec = json.loads(f.read_text())
    assert "http_attempts" in rec["self"]["attempts"][0]
    assert ct.summary(out)["crosscheck_verdicts"].get("not_checked", 0) == 0
    rec["self"]["text"] += " edited after check"; f.write_text(json.dumps(rec))
    sm = ct.summary(out)
    assert sm["crosscheck_verdicts"].get("not_checked") == 1 and sm["all_checked_clean_or_dispute"] is False
    ct.assemble_community(out)
    priv = json.loads((out / "owner_private" / f"{rec['person_id']}.json").read_text())
    assert priv["auto_check_status"] == "not_checked"
