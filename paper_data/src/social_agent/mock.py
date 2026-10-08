"""Deterministic mock for flow tests. Produces schema-valid but content-naive outputs; never a result to report."""
import re


def mock_responder(system: str, user: str, schema):
    n = schema.__name__
    if n == "_Extraction":
        # one required condition per 'This is a must for me.' sentence, attribute guessed from keywords
        req = user.split("CURRENT REQUEST:", 1)[-1]
        conds = []
        for sent in re.split(r"(?<=\.)\s+", req):
            if "must" in sent:
                conds.append({"attribute": "time_window" if "available" in sent else "other", "accepted_values": ["sat_morning"] if "available" in sent else [],
                              "text": sent, "source_span": sent.strip(), "required": True})
        return {"conditions": conds[:4]}
    if n == "_AssessOut":
        ids = re.findall(r"id=(\S+)", user)
        return {"judgements": [{"condition_id": i, "status": "unknown", "evidence_quote": "", "rationale": "mock"} for i in ids],
                "proposed_decision": "insufficient_info"}
    if n == "_QuestionOut":
        return {"question": "Are you available on Saturday morning?"}
    if n == "_DirectCotOut":
        return {"conditions": [{"condition": "x", "required": True, "status": "unknown", "evidence": ""}], "decision": "insufficient_info"}
    if n == "_ScoreOut":
        return {"score": 5, "reasons": "mock"}
    if n == "_RankOut":
        import re as _re
        return {"ranking": _re.findall(r"\[(u\d{4})\]", user)}
    if n == "_DirectOut":
        return {"decision": "insufficient_info", "reasons": "mock"}
    if n == "_SelfFactsText":
        return {"facts": [{"attribute": "time_window", "value": ["sat_morning"], "quote": "Saturday morning"}]}
    if n == "_AnswerOut":
        ids = re.findall(r"\[(\S+?)\]", system)
        return {"answer": "unknown", "evidence_ids": ids[:1], "note": "mock"}
    return {}
