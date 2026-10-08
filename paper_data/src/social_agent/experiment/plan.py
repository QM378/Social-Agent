"""Run the experiments of the paper: every method of every stage in configs/experiments.yaml.

Results go to an output folder (default outputs/), never to records/: records/ holds the published runs.
Each method is its own run (<out>/runs/c_<stage>_<method>/cases.jsonl) and resumes from its own records; candidate lists
are built once per stage and shared by all its methods (<out>/pools/<stage>/). A provenance row per finished method is
appended to <out>/registry.csv. Scoring is a separate step: `social-agent paper-results --records <out>/runs`."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from social_agent.agent import PROMPT_VERSION
from social_agent.community.texts import model_provenance
from social_agent.experiment.runner import EXP_VERSION, data_fingerprint, run_community_experiment
from social_agent.experiment.similarity import run_offline_baselines
from social_agent.llm import CallBudget, load_llm_config, make_client
from social_agent.paths import CONFIGS, resolve

OFFLINE = {"tfidf_top3", "rrf_top3", "graph_ppr_top3"}          # computed from the candidate lists, no model call


def _append_registry(path: Path, row: dict[str, Any]) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


async def run_plan(plan_path: Path, out_root: Path, only_stage: Optional[str] = None, only_method: Optional[str] = None,
                   mock_responder=None) -> list[dict[str, Any]]:
    plan = yaml.safe_load(Path(plan_path).read_text(encoding="utf-8"))
    cdir = resolve(plan["data"]); fp = data_fingerprint(cdir)
    out_root = Path(out_root); results = []
    for stage in plan["stages"]:
        if only_stage and stage["name"] != only_stage:
            continue
        llm = load_llm_config(CONFIGS / stage["model_config"])
        kinds = stage.get("kinds", ["primary", "alternative"]); k = stage.get("pool_k", 8)
        embed = None if mock_responder else stage.get("embed_model", "nomic-embed-text")   # dry run: keyword lists only
        pools_dir = out_root / "pools" / stage["name"]
        prov = {} if mock_responder else await model_provenance(llm)
        for method in stage["methods"]:
            if only_method and method != only_method:
                continue
            run_dir = out_root / "runs" / f"c_{stage['name']}_{method}"
            t0 = datetime.now(timezone.utc)
            client = make_client(llm, budget=CallBudget(None), mock=mock_responder is not None, responder=mock_responder)
            try:
                summary = await run_community_experiment(cdir, run_dir, client, llm, view="partial", methods=[] if method in OFFLINE else [method],
                                                         pool_k=k, embed_model=embed, request_kinds=kinds, subset=None,
                                                         purpose=f"{stage['name']}: {method}", shared_pools_dir=pools_dir)
            finally:
                await client.aclose()
            if method in OFFLINE or method == "llm_score_top3":
                run_offline_baselines(cdir, run_dir, [method])     # llm_score_top3: top three by the recorded scores
            row = {"run_id": run_dir.name, "stage": stage["name"], "method": method, "started_utc": t0.isoformat(),
                   "finished_utc": datetime.now(timezone.utc).isoformat(), "data": plan["data"], "texts_sha256": fp["texts_sha256"],
                   "model": llm.model, "model_digest": prov.get("digest", "unknown"), "model_quant": prov.get("quantization", "unknown"),
                   "temperature": llm.temperature, "seed": llm.seed, "reasoning_effort": llm.reasoning_effort,
                   "prompt_version": PROMPT_VERSION, "exp_version": EXP_VERSION, "pool_k": k, "kinds": ",".join(kinds),
                   "pairs_ok": summary.get("ok"), "execution_errors": summary.get("execution_error"),
                   "requests_sent": summary.get("requests_sent"), "wall_s": summary.get("wall_s")}
            _append_registry(out_root / "registry.csv", row)
            results.append({"run_id": run_dir.name, "pairs_ok": row["pairs_ok"], "execution_errors": row["execution_errors"]})
    return results
