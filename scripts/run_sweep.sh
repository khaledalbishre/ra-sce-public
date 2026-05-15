#!/usr/bin/env bash
set -euo pipefail

LIST_FILE="${1:?Need a list file of yaml paths}"

while IFS= read -r cfg; do
  [[ -z "$cfg" ]] && continue
  [[ "$cfg" =~ ^# ]] && continue

  echo "=================================================="
  echo "RUN: $cfg"
  echo "=================================================="

  python benchmark_ra_sce_v4.py --override_yaml "$cfg"

  RESULTS_DIR=$(grep -E '^results_dir:' "$cfg" | head -n1 | cut -d':' -f2- | xargs)
  DATASET_KEY=$(grep -E '^dataset_key:' "$cfg" | head -n1 | cut -d':' -f2- | xargs)

  if [[ "$DATASET_KEY" == "TREC_DL_19" ]]; then
    QRELS="qrels_dl19.txt"
    RUNFILE="$RESULTS_DIR/run_ra_sce_TREC_DL_19.trec"
  else
    QRELS="qrels_dl20.txt"
    RUNFILE="$RESULTS_DIR/run_ra_sce_TREC_DL_20.trec"
  fi

  echo "EVAL: $RUNFILE"
  python tools/eval_pytrec.py --qrels "$QRELS" --run "$RUNFILE" | tee "$RESULTS_DIR/pytrec_eval.txt"

done < "$LIST_FILE"

echo "✅ Sweep done."
