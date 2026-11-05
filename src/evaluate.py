"""Step 7 - Evaluation harness.

  python src/evaluate.py            -> retrieval metrics only (free, no API key): vector vs BM25 vs hybrid
  python src/evaluate.py --llm      -> also runs the answer step (needs ANTHROPIC_API_KEY, costs a little)
                                       and reports citation validity + refusal accuracy, and writes a
                                       10-question sheet for MANUAL groundedness scoring.

Metrics
  hit@k          answerable question: at least one gold rule is in the top-k retrieved chunks
  all-gold@k     multi-rule question: ALL gold rules are in the top-k
  MRR            mean reciprocal rank of the first gold rule
  gate accuracy  free proxy for refusal: does the similarity gate (answer.MIN_TOP_SIMILARITY)
                 block the unanswerable questions without blocking answerable ones?
"""
import argparse
import csv
import json
import random

import pandas as pd

from config import EVAL_DIR, TOP_K

MODES = ["vector", "bm25", "hybrid"]


def load_questions():
    df = pd.read_csv(EVAL_DIR / "questions.csv", dtype=str).fillna("")
    df["gold"] = df["gold_rules"].apply(lambda s: [g for g in s.split(";") if g])
    return df


def rank_of_gold(results, gold):
    ids = [c["rule_id"] for c in results]
    ranks = [ids.index(g) + 1 for g in gold if g in ids]
    return ranks


def retrieval_eval(r, df, k=TOP_K):
    rows = []
    for mode in MODES:
        hits, allhits, rr, n_ans = 0, 0, [], 0
        multi_all, n_multi = 0, 0
        for _, q in df[df["gold"].str.len() > 0].iterrows():
            res = r.search(q["question"], mode, k)
            ranks = rank_of_gold(res, q["gold"])
            n_ans += 1
            hits += bool(ranks)
            rr.append(1 / min(ranks) if ranks else 0)
            if q["type"] == "multi_rule":
                n_multi += 1
                multi_all += len(ranks) == len(q["gold"])
        rows.append({"retriever": mode, f"hit@{k}": hits / n_ans, "MRR": sum(rr) / n_ans,
                     f"multi all-gold@{k}": multi_all / n_multi if n_multi else float("nan")})
    return pd.DataFrame(rows)


def per_question_misses(r, df, mode, k=TOP_K):
    out = []
    for _, q in df[df["gold"].str.len() > 0].iterrows():
        res = r.search(q["question"], mode, k)
        if len(rank_of_gold(res, q["gold"])) < len(q["gold"]):
            out.append((q["id"], q["question"], q["gold"], [c["rule_id"] for c in res]))
    return out


def gate_sweep(r, df):
    """Top-1 vector similarity for answerable vs unanswerable questions + threshold sweep."""
    ans, un = [], []
    for _, q in df.iterrows():
        top = r.vector(q["question"], 1)[0]["vector_score"]
        (un if q["type"] == "unanswerable" else ans).append(top)
    print(f"\nTop-1 cosine similarity  answerable: min {min(ans):.3f} mean {sum(ans)/len(ans):.3f} | "
          f"unanswerable: max {max(un):.3f} mean {sum(un)/len(un):.3f}")
    print("threshold  answerable_kept  unanswerable_blocked")
    for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        print(f"  {t:.2f}      {sum(a >= t for a in ans)}/{len(ans)}            {sum(u < t for u in un)}/{len(un)}")


def llm_eval(r, df, mode_list=("vector", "hybrid"), k=TOP_K):
    from answer import answer_question
    results = {}
    for mode in mode_list:
        recs = []
        for _, q in df.iterrows():
            chunks = r.search(q["question"], mode, k)
            a = answer_question(q["question"], chunks)
            recs.append({"id": q["id"], "type": q["type"], "question": q["question"], "gold": q["gold"],
                         "status": a.status, "answer": a.text, "citations": a.citations,
                         "invalid": a.invalid_citations, "retrieved": [c["rule_id"] for c in chunks]})
        results[mode] = recs
        answered = [x for x in recs if x["status"] in ("answered", "rejected_invalid_citation")]
        valid = [x for x in answered if x["status"] == "answered"]
        unans = [x for x in recs if x["type"] == "unanswerable"]
        refused_ok = [x for x in unans if x["status"].startswith("refused")]
        ans_q = [x for x in recs if x["type"] != "unanswerable"]
        wrongly_refused = [x for x in ans_q if x["status"].startswith("refused")]
        cites_gold = [x for x in ans_q if x["status"] == "answered" and set(x["gold"]) & set(x["citations"])]
        print(f"\n[{mode}] citation validity: {len(valid)}/{len(answered)} | refusal accuracy (unanswerable): "
              f"{len(refused_ok)}/{len(unans)} | over-refusal (answerable refused): {len(wrongly_refused)}/{len(ans_q)} "
              f"| answered & cites a gold rule: {len(cites_gold)}/{len(ans_q)}")
    EVAL_DIR.joinpath("results").mkdir(exist_ok=True)
    (EVAL_DIR / "results" / "llm_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Manual groundedness sheet: 10 random answered questions from the hybrid run.
    random.seed(7)
    pool = [x for x in results.get("hybrid", []) if x["status"] == "answered"]
    sample = random.sample(pool, min(10, len(pool)))
    with open(EVAL_DIR / "groundedness_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "answer", "cited_rules", "groundedness_0_to_2", "comment"])
        for x in sample:
            w.writerow([x["id"], x["question"], x["answer"], ";".join(x["citations"]), "", ""])
    print("\nWrote eval/groundedness_sample.csv - score each answer 0 (unsupported), 1 (partly), 2 (fully supported by the cited text).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--k", type=int, default=TOP_K)
    args = ap.parse_args()
    from index import load_chunks
    from retrieve import Retriever

    df = load_questions()
    known = {c["rule_id"] for c in load_chunks()}
    bad = [g for gs in df["gold"] for g in gs if g not in known]
    assert not bad, f"Gold rule IDs not found in chunks: {bad}"

    r = Retriever()
    print(f"Questions: {len(df)} ({df['type'].value_counts().to_dict()}), top-k = {args.k}\n")
    table = retrieval_eval(r, df, args.k)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nMisses (hybrid):")
    for m in per_question_misses(r, df, "hybrid", args.k):
        print(f"  Q{m[0]}: {m[1]}\n     gold {m[2]} | got {m[3]}")
    gate_sweep(r, df)
    if args.llm:
        llm_eval(r, df, k=args.k)
    else:
        print("\n(LLM metrics skipped: run `python src/evaluate.py --llm` with ANTHROPIC_API_KEY set.)")


if __name__ == "__main__":
    main()
