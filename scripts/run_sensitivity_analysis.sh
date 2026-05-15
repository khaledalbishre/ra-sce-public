#!/bin/bash
# scripts/run_sensitivity_analysis.sh
# Sensitivity analysis for risk coefficients (Reviewer 2 comment 1)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate venv if available
if [ -f ./venv_bm25/bin/activate ]; then
    source ./venv_bm25/bin/activate
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy-key-not-used}"

OUTPUT_DIR="artifacts/results/sensitivity"
mkdir -p "$OUTPUT_DIR"

echo "================================================"
echo "RA-SCE Coefficient Sensitivity Analysis"
echo "Addresses Reviewer 2 comment 1"
echo "================================================"
echo ""

echo "📊 [1/2] DL19 sensitivity sweep..."
python sensitivity_analysis.py \
    --config configs/ra_sce/dl19_bm25.yaml \
    --output "$OUTPUT_DIR/dl19" \
    2>&1 | tee "$OUTPUT_DIR/dl19_sensitivity_log.txt"

echo ""
echo "📊 [2/2] DL20 sensitivity sweep..."
python sensitivity_analysis.py \
    --config configs/ra_sce/dl20_bm25.yaml \
    --output "$OUTPUT_DIR/dl20" \
    2>&1 | tee "$OUTPUT_DIR/dl20_sensitivity_log.txt"

echo ""
echo "================================================"
echo "✅ Sensitivity analysis complete!"
echo "Results in: $OUTPUT_DIR"
echo "================================================"
