# Provenance

This folder was published on GitHub as `paper_data/` (it was called `research/` while it was assembled; nothing else changed). The paper's final title is *Social Agent: An LLM Agent Framework for Reciprocal Recommendation in Online Dating and Friendship*.

Where each part of this folder comes from, what was checked when it was assembled (2026-10-03), what is known to be imperfect, and what was not checked. No language model was called and no experiment was run while assembling it; the programs run were the unit tests, the offline scoring, and a comparison of old and new code with a mock model (section 2).

## 1. Sources

| Part | Source |
|---|---|
| Method, baselines, candidate lists, run loop, community generator | development version **m4.7**, the version that ran the published experiments (2026-09-24 to 2026-09-26), reorganised into the modules listed in the README; code serving only unreported variants was removed (section 4) |
| Scoring (`evaluation/score.py`, `tables.py`, `reference.py`) | written after the experiments and applied offline to the records; it scores every method against one reference answer, uses only pairs that every method completed, counts the full cost of every recorded call, and breaks down what the conversation changed. It replaces m4.7's own scorer, which compared the 1- and 5-question variants against a different reference than the 3-question variant. |
| `data/community/` | the data snapshot every published run used (status `review_pending`: no later corrections applied); three folders the experiments never read were left out (`full_info_view/`, `owner_private_debug/`, `review/`) |
| `records/` | the published runs: 13 for gpt-oss-20b and 4 for Gemma 4 31B; per run only the raw material (pair outcomes, candidate lists, embeddings, run metadata). Summary files written at the time by m4.7's scorer were left out, so that no second set of metrics exists. `registry.csv` keeps the provenance columns of the original run registry and leaves out its metric columns for the same reason. |

## 2. Checks

**The data is the snapshot the runs used.** The experiment code fingerprints the data by hashing every file in `owner_private/` and `public_directory/directory.jsonl`. For `data/community/` this hash is `6f5c655952dc3232`, the value recorded for all 17 runs in `records/registry.csv`. The reference answers are the original ones.

**Where the records come from.** Fourteen runs were produced by m4.7's batch run. Three (`c_s2_partial_gptoss_direct`, `_B2`, `_B3ft`) contain `REUSED_FROM.txt` naming `exp_e1e2e3_s2_partial`: the batch run copied them from the first experiment round (development version m4.3, 2026-09-24) and then completed a few missing or failed pairs itself on 2026-09-25 (3, 7 and 6 pairs, visible in `finished_utc`). During assembly, the protocol source files of m4.3 and m4.7 were compared and found byte-identical (agent, agent context, question-and-answer control, decision rule, model client, vocabulary); the run loop differed only by added baselines, shared candidate lists and recording of retrieval scores. All runs record prompt version `agent_v5`, temperature 0, seed 0 and the model digests in `registry.csv`.

**The reorganised code behaves like the code that produced the records.** With the same mock model on the real community (30 people and all their requests, 132 pairs), the m4.7 code and this code were run for all thirteen reported methods: 13 x 132 = 1,716 method-pair outcomes. The output files hold 1,848 lines, because `llm_score_top3` writes two lines per pair (the model's raw score, then the offline top-three decision). All 1,848 lines are identical field by field (timestamps and elapsed time excluded), including 116 question-and-answer exchanges, and the candidate lists are identical. This checks the code paths, not the behaviour of a real model.

**The numbers reproduce.** `social-agent paper-results` regenerates the four tables and the budget figure of the paper; their numbers match the paper, and `tests/test_reproduce_paper.py` checks the headline figures (3,289 and 3,307 pairs; two-way recall 45.5% without questions and 78.3% with up to three; 0.56 questions per pair; 189 pairs found, 686 correctly excluded, 8 new wrong handshakes; Gemma 54% to 94%).

**Cases reproduce.** `scripts/export_cases.py` prints the two exchanges quoted in the paper from `records/c_s2_partial_gptoss_B3ft/cases.jsonl`.

**Tests.** `pytest -q`: 40 passed, 3 skipped (they need a live Ollama).

## 3. Known issues in the published records

- **Question metadata carries the asker's wording.** Each question message has a machine-readable note from which the answering program reads the attribute and the acceptable values. In the records this note also contains the asker's wording of that condition (after `; in words:`); all 1,853 question messages of `c_s2_partial_gptoss_B3ft` have it. The answering program does not use that part, so answers and decisions are unaffected, but the messages exchanged were not limited to the yes/no question itself. The code here is the code that produced the records and still writes this note.
- **Two kinds of "can't say" are distinguishable in the notes.** In 23 answers of the same run the note reads "fact exists but is not disclosable" rather than "no stated fact", which tells the asker that the other agent held a value it may not share. No value was disclosed.
- **Texts and settings.** Some generated texts state a wish more or less strongly than the structured settings they were written from. The data was not corrected; the paper states this as a limitation.

## 4. Removed

Variants the paper does not report: the LLM-answer protocol (`B3`), answers from template evidence (`B3f`), LLM re-assessment after each answer, and the full-information view. Early development: the template world generator and its labelling, the template-text pipeline and its checks, matching pilots, model probes, the diagnostic command, and their tests. The old scorer and table builder. Everything from later development that the paper does not use (an app prototype, permission and audit tooling, data adjudication and freezing, unrun baselines). Development reports, scripts and packages. One line in the run loop that passed a field the agent context always dropped was removed; the mock comparison above shows no change in behaviour.

## 5. Not verified

- Re-running the experiments with real models was not done during assembly; whether a rerun on other hardware reproduces the records pair by pair is unknown.
- The records do not store the prompt texts sent; their identity with this code rests on the source comparison and the mock comparison above.
- The Ollama version of the published runs was not recorded; the model digests were.
