#!/bin/bash
# scripts/run_bm25_experiments.sh
# Cross-architecture validation experiments for RA-SCE
# Works on any environment (local or RunPod)

set -e

# Auto-detect project root: script is in scripts/ subdirectory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"

OUTPUT_DIR="artifacts/results/bm25_validation"
mkdir -p "$OUTPUT_DIR"

echo "================================================"
echo "RA-SCE Cross-Architecture Validation"
echo "BM25 (Lucene) + MiniLM-L-12-v2"
echo "================================================"
echo ""

# Phase 1: TREC-DL'19
echo "📊 [1/2] Running TREC-DL'19..."
python benchmark_ra_sce_bm25.py \
    --config configs/ra_sce/dl19_bm25.yaml \
    --output "$OUTPUT_DIR/dl19" \
    2>&1 | tee "$OUTPUT_DIR/dl19_log.txt"

echo ""
echo "📊 [2/2] Running TREC-DL'20..."
python benchmark_ra_sce_bm25.py \
    --config configs/ra_sce/dl20_bm25.yaml \
    --output "$OUTPUT_DIR/dl20" \
    2>&1 | tee "$OUTPUT_DIR/dl20_log.txt"

echo ""
echo "================================================"
echo "✅ All experiments complete!"
echo "Results in: $OUTPUT_DIR"
echo "================================================"

# Quick summary
echo ""
echo "📋 Quick Summary:"
for ds in dl19 dl20; do
    json_file="$OUTPUT_DIR/$ds/${ds}_bm25_results.json"
    if [ -f "$json_file" ]; then
        echo ""
        echo "=== $ds ==="
        python -c "
import json
with open('$json_file') as f:
    r = json.load(f)
agg = r['aggregate_metrics']
print(f\"  Base@k0:    {agg['base'].get('ndcg_cut_10', 0):.4f}\")
print(f\"  CE-All:     {agg['ce_all'].get('ndcg_cut_10', 0):.4f}\")
print(f\"  Random-30%: {agg['random_30_mean'].get('ndcg_cut_10', 0):.4f} ± {agg['random_30_std'].get('ndcg_cut_10', 0):.4f}\")
print(f\"  RA-SCE:     {agg['ra_sce'].get('ndcg_cut_10', 0):.4f}\")
print(f\"  Gating:     {r['gating']['rate']*100:.1f}%\")
"
    fi
done
