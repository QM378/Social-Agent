"""Command line. Run from anywhere; every default path is inside the paper_data/ folder.

  social-agent paper-results              paper tables and numbers from the published records (no model calls)
  social-agent experiments                run the paper's experiments yourself (needs Ollama), into outputs/
  social-agent data <step>                regenerate a community (not needed: the paper's community is in data/)
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from social_agent.paths import CONFIGS, DATA, OUTPUTS, RECORDS, RESULTS, resolve

app = typer.Typer(add_completion=False, no_args_is_help=True)
data_app = typer.Typer(help="Regenerate a simulated community (structured people and requests, LLM texts, cross-model screening).")
app.add_typer(data_app, name="data")


# ----------------------------------------------------------------------------- paper numbers
@app.command("paper-results")
def paper_results(records: str = typer.Option(str(RECORDS), help="folder with c_<stage>_<method>/ runs"),
                  data: str = typer.Option(str(DATA)),
                  stages: str = typer.Option("s2_partial_gptoss,s2_partial_gemma"),
                  out: str = typer.Option(str(RESULTS))):
    """The paper's tables, figure and numbers from recorded runs: audit, common completed pairs, one reference answer for all methods."""
    from social_agent.evaluation.score import build_paper_results
    from social_agent.evaluation.tables import write_tables
    st = [x.strip() for x in stages.split(",")]
    o = resolve(out)
    res = build_paper_results(resolve(data), resolve(records), st, o / "numbers")
    write_tables(res, o / "tables", main_stage=st[0], second_stage=st[1] if len(st) > 1 else "")
    typer.echo(json.dumps({k: {"planned": v["planned"], "common_pairs": v["common_pairs"],
                               "methods": {m: {kk: a[kk] for kk in ("completed_ok", "execution_error", "never_attempted", "extra_not_in_plan")}
                                           for m, a in v["audit"].items()}} for k, v in res.items()}, indent=1))


# ----------------------------------------------------------------------------- experiments
@app.command("experiments")
def experiments(plan: str = typer.Option(str(CONFIGS / "experiments.yaml")), out: str = typer.Option(str(OUTPUTS)),
                stage: str = typer.Option(None, help="run only this stage"), method: str = typer.Option(None, help="run only this method"),
                dry_run: bool = typer.Option(False, help="use a mock model instead of Ollama (checks the plumbing only)")):
    """Run the paper's experiments into an output folder (default outputs/); records/ is never touched. Resumable."""
    from social_agent.experiment.plan import run_plan
    mock = None
    if dry_run:
        from social_agent.mock import mock_responder as mock
    o = resolve(out)
    if o.resolve() == RECORDS.resolve():
        raise typer.BadParameter("records/ holds the published runs; choose another output folder")
    typer.echo(json.dumps(asyncio.run(run_plan(resolve(plan), o, stage, method, mock_responder=mock)), indent=1))


# ----------------------------------------------------------------------------- community data
def _llm(models: str, key: str):
    import yaml
    from social_agent.llm import LLMConfig
    return LLMConfig(**yaml.safe_load(open(resolve(models), encoding="utf-8"))[key])


@data_app.command("generate")
def data_generate(out: str = typer.Option(...), n: int = typer.Option(200), seed: int = typer.Option(2), alt_ratio: float = typer.Option(0.2),
                  models: str = typer.Option("gemma,gptoss"), assign_seed: int = typer.Option(7)):
    """Structured community: people, facts, visibility, requests, reference answers, writer assignment. No model calls.
    The paper's community: --seed 2 --n 200 (defaults)."""
    from collections import Counter
    from social_agent.community.generate import export_structured, generate_community
    from social_agent.community.texts import assign_models
    from social_agent.vocab import load_vocab
    o = resolve(out); v = load_vocab()
    comm = generate_community(v, seed=seed, n_people=n, alt_ratio=alt_ratio)
    m = export_structured(v, comm, o)
    a = assign_models(comm, [k.strip() for k in models.split(",")], assign_seed)
    (o / "generation" / "assignment.json").write_text(json.dumps(a, indent=1), encoding="utf-8")
    typer.echo(json.dumps({"manifest": m, "assignment_counts": dict(Counter(a.values())), "report": comm["report"]}, indent=2))


@data_app.command("write-texts")
def data_write_texts(out: str = typer.Option(...), model_key: str = typer.Option(...), models: str = typer.Option(str(CONFIGS / "text_models.yaml"))):
    """The assigned LLM writes self-descriptions and requests from the structured notes."""
    from social_agent.community.texts import render_community
    typer.echo(json.dumps(asyncio.run(render_community(resolve(out), model_key, _llm(models, model_key))), indent=1))


@data_app.command("crosscheck")
def data_crosscheck(out: str = typer.Option(...), checker_key: str = typer.Option(...), models: str = typer.Option(str(CONFIGS / "text_models.yaml"))):
    """The other model reads the texts back and flags differences from the structured notes (never rewrites)."""
    from social_agent.community.texts import crosscheck_community
    typer.echo(json.dumps(asyncio.run(crosscheck_community(resolve(out), checker_key, _llm(models, checker_key))), indent=1))


@data_app.command("recheck")
def data_recheck(out: str = typer.Option(...)):
    """Recompute the screening verdicts from the stored checker responses (no model calls)."""
    from social_agent.community.texts import recheck_offline
    typer.echo(json.dumps(recheck_offline(resolve(out)), indent=2))


@data_app.command("assemble")
def data_assemble(out: str = typer.Option(...)):
    """Write the layers agents read: owner_private/ (own texts, disclosure policy) and public_directory/ (public profiles)."""
    from social_agent.community.texts import assemble_community, summary
    o = resolve(out)
    m = assemble_community(o); sm = summary(o)
    (o / "summary.json").write_text(json.dumps({"manifest": m, "summary": sm}, indent=1), encoding="utf-8")
    typer.echo(json.dumps({"manifest": m, "summary": sm}, indent=2))
