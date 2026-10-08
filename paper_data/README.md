# paper_data: code, data and records of the Social Agent paper

This folder accompanies the paper *Social Agent: An LLM Agent Framework for Reciprocal Recommendation in Online Dating and Friendship* (Qiming Guo, Jinwen Tang, Xingran Huang, Hung-Yu Lin, Yafu Zhong, Xiatian Zhuang). It holds everything the paper's results rest on:

- **the simulated community** the experiments use (200 people, 240 requests), already generated: `data/community/`;
- **the method** and **the baselines** it is compared with: `src/social_agent/`;
- **the recorded runs**, one per method and model, holding every pair outcome reported in the paper: `records/`;
- **the scoring** that turns those records into every table, figure and number of the paper.

This is research code, not an application. There is no interface, no accounts and no deployment code.

All paths are relative to this `paper_data/` folder; commands can be run from any directory.

## Setup

Python 3.11 or 3.12.

```bash
cd paper_data
pip install -e ".[dev]"        # ".[dev]" adds pytest for the tests
pytest -q                      # 40 tests; 3 more need a live Ollama and are skipped
```

## 1. Reproduce the paper's numbers (no GPU, no model calls)

```bash
social-agent paper-results
```

This reads `records/`, keeps only the pairs that every method completed (3,289 pairs for gpt-oss-20b, 3,307 for Gemma 4 31B), scores every method against the same reference answers, counts the full cost of every recorded call, and writes:

| Output | Content |
|---|---|
| `results/tables/tab_directional.tex` | one-sided judgments of all methods |
| `results/tables/tab_introduction.tex` | two-way handshakes and their cost |
| `results/tables/tab_clarify.tex` | what the conversation changed |
| `results/tables/tab_clarify_full.tex` | every kind of change, summing to all pairs |
| `results/tables/fig_budget.tex` | handshakes found and their precision by question budget |
| `results/numbers/*.json` | per-method audit, the common pair list, all scores (the numbers quoted in the text) |

The printed audit shows, for each method, how many planned pairs were completed, failed, never attempted, or present in the records but outside the plan. This command is the only scoring entry point; nothing else in the repository computes metrics.

The two conversations quoted in the paper come straight from the records:

```bash
python scripts/export_cases.py
```

The paper writes names instead of person ids and plain words instead of code labels (recommend = agree, reject = disagree, insufficient_info or unknown = not sure or can't say).

## 2. Run the experiments yourself (GPU and Ollama)

```bash
ollama pull gpt-oss:20b
ollama pull gemma4:31b
ollama pull nomic-embed-text
ollama create social-gptoss-8k -f configs/ollama/Modelfile.gptoss-8k
ollama create social-gemma4-8k -f configs/ollama/Modelfile.gemma4-8k

social-agent experiments                                   # every stage and method of the paper, into outputs/
social-agent experiments --stage s2_partial_gptoss --method B3ft   # one method
social-agent paper-results --records outputs/runs --out outputs/results
```

Runs are written to `outputs/` (or `--out`), never to `records/`; the command refuses `records/` as an output. A run resumes from its own records if interrupted. `--dry-run` replaces the model with a fixed mock to check the plumbing in seconds; its results mean nothing.

Model calls use temperature 0 and seed 0, but language model outputs can differ across hardware, drivers and Ollama versions, so a rerun will be close to, not identical with, the published records. Edit `configs/matcher_*.yaml` if Ollama is not at `http://127.0.0.1:11434`.

## 3. The community data

`data/community/` is the community every published run used. It was generated once and is not regenerated:

| Folder | Content | Read by agents? |
|---|---|---|
| `owner_private/` | each person's self-description, requests and disclosure settings (what that person's own agent may read) | own agent only |
| `public_directory/` | public profiles | yes |
| `evaluator/` | structured facts, requests and the reference compatibility matrices | never |
| `generation/` | generation notes, the LLM-written texts with their attempts, cross-model screening, writer provenance | never |

How it was made: a program draws 200 people over three activities (street photography, online gaming, study partners), with facts, a visibility for each fact (public, shared when asked, never shared) and requests with must-haves and nice-to-haves; 40 people get a second request that changes one thing. Gemma 4 31B and gpt-oss-20b each wrote half of the self-descriptions and requests from these facts, and the other model read each text back to flag differences. Because the facts are known, a program computes for every pair whether each person's must-haves hold: these are the reference answers.

The generator is included (`social-agent data generate | write-texts | crosscheck | recheck | assemble`, with `--seed 2 --n 200` for this community). Structured people, requests and reference answers come out the same for the same seed; the texts are written at temperature 0.7 and would differ.

## 4. How the method works in the code

| Step | Who does it | Code |
|---|---|---|
| Turn the owner's request into must-haves and nice-to-haves, each tied to the words it came from | LLM | `agent.py`: `PersonalAgent.extract` |
| Read the other person's public profile: each wish met, not met, or not stated, with a quote | LLM, quotes checked by the program | `agent.py`: `assess` |
| Extract the owner's own facts from the owner's own texts, each with a verbatim quote | LLM, quotes checked by the program | `agent.py`: `extract_self_facts_from_text` |
| Pick the next open must-have, alternating sides, within the question budget | program | `protocol.py`: `run_pair`, `agent.py`: `next_unresolved` |
| Phrase one yes/no question about it | LLM | `agent.py`: `ask` |
| Answer yes, no or can't say from the owner's extracted facts and disclosure settings | program | `agent.py`: `answer_from_facts` |
| Update only the asked condition | program | `agent.py`: `update_from_answers` |
| Decide each side (any must-have not met: no; any unknown: not sure; otherwise yes) and the handshake (both yes) | program | `schemas.py`: `decide_direction`, `aggregate_joint` |

So the exchange between agents is a constrained question-and-answer protocol: the LLM understands language and writes questions; answers, updates and decisions are computed by the program from what the LLM extracted.

Other parts:

| | Code |
|---|---|
| Candidate list per request: top 8 by BM25 plus top 8 by embedding over public profiles of the same activity | `experiment/runner.py`: `build_pools` |
| LLM baselines (direct judgment, conditions first, 0 to 9 score, listwise ranking) and the per-pair run loop | `experiment/runner.py` |
| Similarity baselines (TF-IDF, rank fusion, keyword graph) | `experiment/similarity.py` |
| Batch runs from `configs/experiments.yaml` | `experiment/plan.py` |
| Reference answers, scores, common pair set, tables | `evaluation/reference.py`, `score.py`, `tables.py` |
| Community generation | `community/generate.py` (people, requests, reference matrices, template notes and public profiles), `community/texts.py` (LLM texts, screening, assembly) |
| Vocabulary and structured types | `vocab.py`, `schemas.py` |
| Model client, paths, mock model | `llm.py`, `paths.py`, `mock.py` |

Method ids:

| Id | In the paper |
|---|---|
| `bm25_top3`, `tfidf_top3`, `embed_top3`, `rrf_top3`, `graph_ppr_top3` | similarity search (keyword, TF-IDF, embedding, combined, keyword graph), top three |
| `direct`, `direct_cot` | an LLM judges the request against the profile; the same, listing the wishes first |
| `llm_score_top3`, `llm_rank_top3` | an LLM scores or ranks the candidates |
| `B2` | Social Agent without conversation |
| `B3ft_q1`, `B3ft`, `B3ft_q5` | Social Agent with up to 1, 3 or 5 questions per pair |

## 5. Layout

```
paper_data/
  configs/
    experiments.yaml      the paper's experiments: two stages (gpt-oss-20b, Gemma 4 31B) and their methods
    matcher_gptoss.yaml   model settings of the recorded gpt-oss-20b runs
    matcher_gemma.yaml    model settings of the recorded Gemma 4 31B runs
    text_models.yaml      the two models that wrote and screened the community texts
    vocab.yaml            attribute vocabulary used by the community and the agents
    ollama/               Modelfiles for the 8k-context aliases
  data/community/         the community (section 3)
  records/
    c_<stage>_<method>/   one published run: cases.jsonl (every pair outcome), pools.json, embeddings.json, run_meta.json
    registry.csv          provenance of each published run (data hash, model digest, versions)
  results/                generated by `social-agent paper-results`
  scripts/export_cases.py print recorded exchanges verbatim
  src/social_agent/       the code (section 4)
  tests/                  unit tests, the method on the real community with a mock model, and reproduction of the paper's numbers
  PROVENANCE.md           where each part comes from, what was checked, known issues
```

## 6. License

Code (everything except `data/` and `records/`) is released under the MIT License, see `../LICENSE`. The simulated community in `data/` and the run records in `records/` are released under CC BY 4.0, see `data/LICENSE-DATA.md`. All people, profiles and requests in the data are synthetic.

## 7. Citation

If you use this code or data, please cite the paper; see `../CITATION.cff`.
