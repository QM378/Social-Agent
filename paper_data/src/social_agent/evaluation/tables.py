"""LaTeX tables for the paper, generated ONLY from build_paper_results() output (no hand-copied numbers)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

NAMES = {"bm25_top3": "BM25 (top-3)", "tfidf_top3": "TF-IDF (top-3)", "embed_top3": "Dense embedding (top-3)", "rrf_top3": "Hybrid RRF (top-3)",
         "graph_ppr_top3": "Keyword-graph PPR (top-3)", "llm_score_top3": "LLM 0--9 scorer (top-3)", "llm_rank_top3": "LLM listwise rerank (top-3)",
         "direct": "Direct LLM", "direct_cot": "Direct LLM, conditions first",
         "B2": "Social Agent, no clarification", "B3ft_q1": "Social Agent, $\\le$1 question", "B3ft": "Social Agent, $\\le$3 questions", "B3ft_q5": "Social Agent, $\\le$5 questions"}
GROUPS = [("Retrieval and graph", ["bm25_top3", "tfidf_top3", "embed_top3", "rrf_top3", "graph_ppr_top3"]),
          ("Single LLM", ["llm_score_top3", "llm_rank_top3", "direct", "direct_cot"]), ("Ours", ["B2", "B3ft_q1", "B3ft", "B3ft_q5"])]


def _f(x):
    return "--" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def write_tables(res: dict[str, Any], out: Path, main_stage: str = "s2_partial_gptoss", second_stage: str = "s2_partial_gemma") -> None:
    out.mkdir(parents=True, exist_ok=True)
    P = res[main_stage]["scores"]; n_common = res[main_stage]["common_pairs"]
    t = ["\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{2.5pt}\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{lccccc}\\toprule",
         "Method & Acc$_3$ & P & WIR$\\downarrow$ & R & UnkR \\\\ \\midrule"]
    for g, ms in GROUPS:
        t.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{g}}}}}\\\\")
        for m in ms:
            v = P["A_directional_disclosable"].get(m)
            if not v:
                continue
            rank = m.endswith("top3")
            t.append(f"{NAMES[m]} & {'--' if rank else _f(v['acc3'])} & {_f(v['P'])} & {_f(v['WIR'])} & {_f(v['R'])} & {'--' if rank else _f(v['InsRec'])} \\\\")
    t += ["\\bottomrule\\end{tabular}}",
          f"\\caption{{Directional judgment on the common completed set ({n_common:,} requester--candidate pairs completed by every method; matcher gpt-oss-20b; partial view). "
          "All methods are scored against the same target: whether the requester's current request accepts the candidate given what the candidate would truthfully disclose. "
          "P: precision of recommendations; WIR: share of recommendations violating a required condition; R: recall of acceptable candidates; UnkR: recall of undetermined pairs. "
          "Rank-based methods recommend the top three and never reject.}", "\\label{tab:directional}\\end{table}"]
    (out / "tab_directional.tex").write_text("\n".join(t), encoding="utf-8")
    t = ["\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{2.5pt}\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{lccccc}\\toprule",
         "Method & Acc$_3$ & P & R & calls & tokens \\\\ \\midrule"]
    for stage, label in ((main_stage, "gpt-oss-20b"), (second_stage, "Gemma 4 31B")):
        if stage not in res:
            continue
        S = res[stage]["scores"]; t.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{label}, partial view ({res[stage]['common_pairs']:,} pairs)}}}}\\\\")
        for m in ["B2", "B3ft_q1", "B3ft", "B3ft_q5"]:
            v = S["B_introduction"].get(m); c = S["cost"].get(m)
            if v:
                t.append(f"{NAMES[m]} & {_f(v['acc3'])} & {_f(v['P'])} & {_f(v['R'])} & {c['calls_per_pair']:.2f} & {c['tokens_per_pair']} \\\\")
    t += ["\\bottomrule\\end{tabular}}", "\\caption{Two-way introduction decisions on the common completed set. Reference: both directions acceptable under truthful disclosure. "
          "Calls and tokens per pair count every LLM call of the method, including failed and retried calls.}", "\\label{tab:introduction}\\end{table}"]
    (out / "tab_introduction.tex").write_text("\n".join(t), encoding="utf-8")
    # main-text clarification table: what asking changes, in plain terms
    t = ["\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{lrrrr}\\toprule",
         "Budget & found & excluded & new wrong & q/pair \\\\ \\midrule"]
    full = ["\\begin{table*}[t]\\centering\\small\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lrrrrrrrrrr}\\toprule",
            "Budget & pairs & unchanged & fixed: found & fixed: excl. & fixed: other & broken: new wrong & broken: lost & broken: other & changed, still wrong & check \\\\ \\midrule"]
    for stage, suffix in ((main_stage, ""), (second_stage, " (Gemma)")):
        if stage not in res:
            continue
        S = res[stage]["scores"]
        for k, lab in (("B2->B3ft_q1", "$\\le$1"), ("B2->B3ft", "$\\le$3"), ("B2->B3ft_q5", "$\\le$5")):
            e = S["E3"].get(k)
            if not e:
                continue
            found = e.get("fixed:recovered_introduction", 0); excl = e.get("fixed:now_correct_reject", 0)
            other = e.get("fixed", 0) - found - excl
            nw = e.get("broken:new_wrong_introduction", 0); lost = e.get("broken:lost_introduction", 0); bo = e.get("broken", 0) - nw - lost
            q = S["cost"][k.split("->")[1]]["questions_per_pair"]
            t.append(f"{lab}{suffix} & {found} & {excl} & {nw} & {q:.2f} \\\\")
            total = e.get("unchanged", 0) + e.get("fixed", 0) + e.get("broken", 0) + e.get("changed_still_wrong", 0)
            full.append(f"{lab}{suffix} & {e['pairs']} & {e.get('unchanged',0)} & {found} & {excl} & {other} & {nw} & {lost} & {bo} & {e.get('changed_still_wrong',0)} & {'ok' if total == e['pairs'] else 'MISMATCH'} \\\\")
    t += ["\\bottomrule\\end{tabular}", "\\caption{What clarification changes, relative to the same protocol without questions, on identical pairs. "
          "\\emph{found}: an acceptable pair that was undecided is now introduced; \\emph{excluded}: an undecided pair is now correctly excluded; "
          "\\emph{new wrong}: a pair that was not introduced is now wrongly introduced; q/pair: questions per pair averaged over all pairs, including pairs that needed none. "
          "All categories, summing to the number of pairs, are in Table~\\ref{tab:clarify-full}.}", "\\label{tab:clarify}\\end{table}"]
    full += ["\\bottomrule\\end{tabular}}", "\\caption{All outcome changes from clarification. \\emph{check}: the categories sum to the number of pairs.}", "\\label{tab:clarify-full}\\end{table*}"]
    (out / "tab_clarify.tex").write_text("\n".join(t), encoding="utf-8")
    (out / "tab_clarify_full.tex").write_text("\n".join(full), encoding="utf-8")
    # budget curve, generated from the same results
    B = P["B_introduction"]; pts = [(0, "B2"), (1, "B3ft_q1"), (3, "B3ft"), (5, "B3ft_q5")]
    rec = " ".join(f"({b},{B[m]['R']:.3f})" for b, m in pts if m in B); pre = " ".join(f"({b},{B[m]['P']:.3f})" for b, m in pts if m in B)
    fig = ["\\begin{figure}[t]", "\\centering", "\\begin{tikzpicture}",
           "\\begin{axis}[width=0.95\\columnwidth, height=4.2cm, xlabel={question budget per pair}, ylabel={two-way score}, xtick={0,1,3,5}, ymin=0.4, ymax=1.0,",
           "legend style={at={(0.97,0.05)}, anchor=south east, font=\\footnotesize}, tick label style={font=\\footnotesize}, label style={font=\\footnotesize}, grid=major]",
           f"\\addplot[mark=*, thick] coordinates {{{rec}}};", "\\addlegendentry{recall (within pool)}",
           f"\\addplot[mark=square*, thick, dashed] coordinates {{{pre}}};", "\\addlegendentry{precision}",
           "\\end{axis}", "\\end{tikzpicture}",
           f"\\caption{{Two-way introductions as the question budget grows (gpt-oss-20b, {n_common:,} pairs). The first question gives most of the gain; precision does not drop.}}",
           "\\label{fig:budget}", "\\end{figure}"]
    (out / "fig_budget.tex").write_text("\n".join(fig), encoding="utf-8")
    (out / "numbers.json").write_text(json.dumps({k: {"common_pairs": v["common_pairs"], "scores": v["scores"]} for k, v in res.items()}, indent=1), encoding="utf-8")
