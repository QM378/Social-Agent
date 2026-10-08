"""Unified evaluation (scorer v3). One task target for every method, computed from raw cases.jsonl.

Table A (directional task, all methods): does A's request accept candidate B?  prediction = decision_ab
  (rank baselines: top-3 = recommend). Reference = status_disclosable[i][j] (what is knowable if B answered truthfully);
  secondary reference = status_public[i][j] (what is decidable from the public card alone).
Table B (introduction task, bidirectional methods only): introduce the pair?  prediction = joint decision.
  Reference = joint(status_disclosable[i][j], status_disclosable[j][i]); switch requests use the switch row/column.
E2: for each alternative request pair, joint reference before/after and whether the AFTER prediction is correct.
E3: B2 vs each B3 variant on the intersection of pairs; counts sum to that intersection (asserted).
Cost: summed over ALL raw records of the method (offline finalisation rows never hide the original LLM calls).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from social_agent.evaluation.reference import _cell, _joint, _reverse_cell

BIDIR = {"B2", "B3ft", "B3ft_q1", "B3ft_q5"}


def _raw(run_dir: Path, method: str) -> tuple[dict, Counter]:
    latest: dict[tuple, dict] = {}; cost = Counter()
    for line in (run_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["method"] != method:
            continue
        u = d.get("usage") or {}
        if not d.get("offline"):
            cost["calls"] += u.get("calls", 0) or 0; cost["tokens"] += (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)
        k = (d["request_id"], d["candidate_id"])
        if k not in latest or d.get("execution_status") == "ok":
            latest[k] = d
    return latest, cost


def _pred_dir(c: dict) -> str:
    if c.get("execution_status") != "ok":
        return "execution_error"
    return c.get("decision_ab") or "execution_error"


def _metrics(pairs: list[tuple[str, str, bool]]) -> dict[str, Any]:
    """pairs: (pred, ref, required_conflict_on_recommend)"""
    n = len(pairs); acc = sum(p == r for p, r, _ in pairs)
    rec = [(p, r, w) for p, r, w in pairs if p == "recommend"]
    ref_rec = [p for p, r, _ in pairs if r == "recommend"]
    ref_ins = [p for p, r, _ in pairs if r == "insufficient_info"]
    return {"n": n, "acc3": round(acc / n, 3) if n else None, "n_rec": len(rec),
            "P": round(sum(r == "recommend" for _, r, _ in rec) / len(rec), 3) if rec else None,
            "WIR": round(sum(w for _, _, w in rec) / len(rec), 3) if rec else None,
            "R": round(sum(p == "recommend" for p in ref_rec) / len(ref_rec), 3) if ref_rec else None,
            "InsRec": round(sum(p == "insufficient_info" for p in ref_ins) / len(ref_ins), 3) if ref_ins else None,
            "errors": sum(p == "execution_error" for p, _, _ in pairs)}


def score(cdir: Path, runs: dict[str, Path]) -> dict[str, Any]:
    m = json.loads((cdir / "evaluator" / "matrices.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {"A_directional_disclosable": {}, "A_directional_public": {}, "B_introduction": {}, "cost": {}, "E2": {}, "E3": {}}
    cache = {}
    for meth, rd in runs.items():
        latest, cost = _raw(rd, meth); cache[meth] = latest
        rows_d, rows_p, rows_b = [], [], []
        for (rid, j), c in latest.items():
            cell = _cell(m, rid, c["person_id"], j, c["request_kind"])
            if not cell or cell.get("different_activity"):
                continue
            pd = _pred_dir(c)
            rows_d.append((pd, cell["status_disclosable"], bool(cell["required_conflict"])))
            rows_p.append((pd, cell["status_public"], bool(cell["required_conflict"])))
            if meth in BIDIR:
                rc = _reverse_cell(m, rid, c["person_id"], j, c["request_kind"])
                if rc:
                    pj = c.get("joint_decision") if c.get("execution_status") == "ok" else "execution_error"
                    rows_b.append((pj or "execution_error", _joint(cell["status_disclosable"], rc["status_disclosable"]),
                                   bool(cell["required_conflict"] or rc.get("required_conflict"))))
        out["A_directional_disclosable"][meth] = _metrics(rows_d)
        out["A_directional_public"][meth] = _metrics(rows_p)
        if rows_b:
            out["B_introduction"][meth] = _metrics(rows_b)
        n = len(rows_d)
        out["cost"][meth] = {"calls_per_pair": round(cost["calls"] / n, 2) if n else None, "tokens_per_pair": round(cost["tokens"] / n) if n else None,
                             "questions_per_pair": round(sum((c.get("questions_used") or 0) for c in latest.values()) / n, 2) if n else None}
    # E3 on the exact intersection
    if "B2" in cache:
        for v in [x for x in ("B3ft_q1", "B3ft", "B3ft_q5") if x in cache]:
            keys = sorted(set(cache["B2"]) & set(cache[v])); cnt = Counter(); used = 0
            for k in keys:
                c2, c3 = cache["B2"][k], cache[v][k]
                cell = _cell(m, k[0], c3["person_id"], k[1], c3["request_kind"]); rc = _reverse_cell(m, k[0], c3["person_id"], k[1], c3["request_kind"])
                if not cell or cell.get("different_activity") or not rc:
                    continue
                used += 1
                ref = _joint(cell["status_disclosable"], rc["status_disclosable"])
                p2 = c2.get("joint_decision") if c2.get("execution_status") == "ok" else "execution_error"
                p3 = c3.get("joint_decision") if c3.get("execution_status") == "ok" else "execution_error"
                kind = "unchanged" if p2 == p3 else "fixed" if p3 == ref else "broken" if p2 == ref else "changed_still_wrong"
                cnt[kind] += 1
                if kind == "fixed":     # what kind of correction
                    cnt["fixed:recovered_introduction" if ref == "recommend" else
                        "fixed:removed_wrong_introduction" if p2 == "recommend" else
                        "fixed:now_correct_reject" if ref == "reject" else "fixed:now_correct_undetermined"] += 1
                if kind == "broken":
                    cnt["broken:lost_introduction" if p2 == "recommend" == ref else
                        "broken:new_wrong_introduction" if p3 == "recommend" else "broken:other"] += 1
            assert sum(v for k, v in cnt.items() if ":" not in k) == used
            out["E3"][f"B2->{v}"] = {"pairs": used, **dict(cnt)}
    # E2: alternative requests, joint reference before/after, correctness after
    for meth in [x for x in cache if x in BIDIR]:
        prim = {(c["person_id"], j): c for (rid, j), c in cache[meth].items() if c["request_kind"] == "primary"}
        cnt = Counter()
        for (rid, j), c in cache[meth].items():
            if c["request_kind"] != "alternative":
                continue
            p0 = prim.get((c["person_id"], j))
            sw = m["switches"].get(rid)
            if not p0 or not sw or j not in sw["row"] or j not in sw["column"]:
                continue
            before = _joint(m["main"][c["person_id"]][j]["status_disclosable"], m["main"][j][c["person_id"]]["status_disclosable"])
            after = _joint(sw["row"][j]["status_disclosable"], sw["column"][j]["status_disclosable"])
            pa = c.get("joint_decision") if c.get("execution_status") == "ok" else "execution_error"
            cnt[("ref_changed" if before != after else "ref_same") + ("|correct_after" if pa == after else "|wrong_after")] += 1
        if cnt:
            out["E2"][meth] = dict(cnt)
    return out


def to_markdown(res: dict[str, Any], title: str) -> str:
    def tab(name, d, cols):
        s = f"\n### {name}\n\n| method | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n"
        for k, v in d.items():
            s += f"| {k} | " + " | ".join(str(v.get(c)) for c in cols) + " |\n"
        return s
    cols = ["n", "acc3", "P", "WIR", "R", "InsRec", "n_rec", "errors"]
    md = f"## {title}\n"
    md += tab("A. Directional task, reference = disclosable (same target for all methods)", res["A_directional_disclosable"], cols)
    md += tab("A'. Directional task, reference = public card only", res["A_directional_public"], cols)
    md += tab("B. Introduction task (bidirectional methods), reference = joint disclosable", res["B_introduction"], cols)
    md += tab("Cost (raw LLM calls, offline rows excluded)", res["cost"], ["calls_per_pair", "tokens_per_pair", "questions_per_pair"])
    md += tab("E3 clarification effect on the exact pair intersection", res["E3"], ["pairs", "fixed", "broken", "changed_still_wrong", "unchanged"])
    md += tab("E2 request switch (joint reference before/after; correctness after the switch)", res["E2"],
              ["ref_changed|correct_after", "ref_changed|wrong_after", "ref_same|correct_after", "ref_same|wrong_after"])
    return md


# ----------------------------------------------------------------------------- common completed set and paper results
# Common-completed-set evaluation (scorer v3.1).
# Planned pairs = the shared pool file of the stage (primary pool reused for alternative requests).
# For each method: planned, completed ok, execution errors, never attempted, extra (not in plan).
# Main comparison = pairs completed ok by EVERY listed method; counts are reported alongside.
def planned_pairs(pool_file: Path) -> set[tuple[str, str]]:
    pools = json.loads(pool_file.read_text(encoding="utf-8"))["pools"]
    prim = {v["person_id"]: v["pool"] for k, v in pools.items() if k.endswith(".r1")}
    out = set()
    for rid, v in pools.items():
        pool = v["pool"] if rid.endswith(".r1") else prim.get(v["person_id"], v["pool"])
        out |= {(rid, j) for j in pool}
    return out


def audit(runs: dict[str, Path], plan: set) -> tuple[dict[str, Any], set]:
    rep = {}; common = None
    for meth, rd in runs.items():
        latest, _ = _raw(rd, meth)
        ok = {k for k, c in latest.items() if c.get("execution_status") == "ok"}
        err = {k for k, c in latest.items() if c.get("execution_status") != "ok"}
        rep[meth] = {"planned": len(plan), "completed_ok": len(ok & plan), "execution_error": len(err & plan),
                     "never_attempted": len(plan - set(latest)), "extra_not_in_plan": len(set(latest) - plan),
                     "missing_examples": sorted(plan - ok)[:5]}
        common = (ok & plan) if common is None else common & ok
    return rep, common or set()


def filter_runs(runs: dict[str, Path], keep: set, tmp: Path) -> dict[str, Path]:
    out = {}
    for meth, rd in runs.items():
        d = tmp / meth; d.mkdir(parents=True, exist_ok=True)
        lines = [l for l in (rd / "cases.jsonl").read_text(encoding="utf-8").splitlines()
                 if l.strip() and (json.loads(l)["request_id"], json.loads(l)["candidate_id"]) in keep]
        (d / "cases.jsonl").write_text("\n".join(lines), encoding="utf-8")
        out[meth] = d
    return out


def build_paper_results(cdir: Path, runs_root: Path, stages: list[str], out_dir: Path, pool_method: str = "B3ft") -> dict[str, Any]:
    """Single official path for paper numbers. For each stage:
    plan (shared pool) -> per-method audit -> common completed set -> scores on that set -> cost on ALL records.
    Writes out_dir/<stage>_audit.json, <stage>_common_pairs.json, <stage>_scores.json and returns everything."""
    import shutil
    out_dir.mkdir(parents=True, exist_ok=True); allres = {}
    for stage in stages:
        runs = {p.name.split(f"c_{stage}_", 1)[1]: p for p in sorted(runs_root.glob(f"c_{stage}_*"))}
        if not runs:
            continue
        plan = planned_pairs(runs[pool_method if pool_method in runs else next(iter(runs))] / "pools.json")
        rep, common = audit(runs, plan)
        tmp = out_dir / f"_filtered_{stage}"; shutil.rmtree(tmp, ignore_errors=True)
        scored = score(cdir, filter_runs(runs, common, tmp))
        full_cost = score(cdir, runs)["cost"]                 # cost over every record, failures and retries included
        scored["cost"] = full_cost
        shutil.rmtree(tmp, ignore_errors=True)
        (out_dir / f"{stage}_audit.json").write_text(json.dumps(rep, indent=1, default=list), encoding="utf-8")
        (out_dir / f"{stage}_common_pairs.json").write_text(json.dumps(sorted(common)), encoding="utf-8")
        (out_dir / f"{stage}_scores.json").write_text(json.dumps(scored, indent=1), encoding="utf-8")
        allres[stage] = {"planned": len(plan), "common_pairs": len(common), "audit": rep, "scores": scored}
    (out_dir / "paper_results.json").write_text(json.dumps(allres, indent=1, default=list), encoding="utf-8")
    return allres
