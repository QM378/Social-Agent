"""The published records reproduce the paper's headline numbers (offline scoring, about a minute)."""
from social_agent.evaluation.score import build_paper_results
from social_agent.paths import DATA, RECORDS


def test_paper_numbers(tmp_path):
    r = build_paper_results(DATA, RECORDS, ["s2_partial_gptoss", "s2_partial_gemma"], tmp_path)
    P, G = r["s2_partial_gptoss"], r["s2_partial_gemma"]
    assert (P["common_pairs"], G["common_pairs"]) == (3289, 3307)
    B = P["scores"]["B_introduction"]
    assert (round(B["B2"]["R"] * 100, 1), round(B["B3ft"]["R"] * 100, 1)) == (45.5, 78.3)
    assert P["scores"]["cost"]["B3ft"]["questions_per_pair"] == 0.56
    e = P["scores"]["E3"]["B2->B3ft"]
    assert (e["fixed:recovered_introduction"], e["fixed:now_correct_reject"], e["broken:new_wrong_introduction"]) == (189, 686, 8)
    assert (round(G["scores"]["B_introduction"]["B2"]["R"] * 100), round(G["scores"]["B_introduction"]["B3ft"]["R"] * 100)) == (54, 94)
