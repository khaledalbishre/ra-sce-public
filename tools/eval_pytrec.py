import argparse
import pytrec_eval

def load_qrels(path):
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            if len(parts) == 4:
                # Standard TREC format: qid 0 docid rel
                qid, _, docid, rel = parts
            elif len(parts) == 3:
                # 3-column format: qid docid rel
                qid, docid, rel = parts
            else:
                raise ValueError(f"Unexpected qrels format: {line}")

            qrels.setdefault(qid, {})[docid] = int(rel)

    return qrels


def load_run(path):
    run = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, _, docid, _, score, _ = line.split()
            run.setdefault(qid, {})[docid] = float(score)
    return run

def mrr_at_k(run, qrels, k=10):
    rr = []
    for qid, docs in run.items():
        ranked = sorted(docs.items(), key=lambda x: x[1], reverse=True)[:k]
        val = 0.0
        for i, (docid, _) in enumerate(ranked, start=1):
            if qrels.get(qid, {}).get(docid, 0) > 0:  # rel > 0 (binary relevance)
                val = 1.0 / i
                break
        if qid in qrels and any(v > 0 for v in qrels[qid].values()):
            rr.append(val)
    return sum(rr) / max(1, len(rr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    qrels = load_qrels(args.qrels)
    run = load_run(args.run)

    emetrics = {
        "ndcg_cut.10",
        "ndcg_cut.20",
        "map_cut.1000",
        "recip_rank",
        "recall.100",
        "recall.1000"
    }

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, emetrics)
    res = evaluator.evaluate(run)

    print("\n=== Mean Metrics (pytrec_eval) ===")
    for m in emetrics:
        key = m.replace(".", "_")  # output keys use underscores
        vals = [v.get(key, 0.0) for v in res.values()]
        mean = sum(vals) / max(1, len(vals))
        print(f"{m}: {mean:.6f}")

    print(f"mrr@10 (manual cutoff): {mrr_at_k(run, qrels, k=10):.6f}")

if __name__ == "__main__":
    main()
