#!/bin/bash
# Convenience script to analyze ranking performance for positives and negatives

PREDICTIONS_FILE="${1:-data/2025-09-25T18-55_export - 2025-09-25T18-55_export.csv}"
POSITIVES_FILE="${2:-data/positives.csv}"
NEGATIVES_FILE="${3:-data/negatives.csv}"
MATCH_FIELD="${4:-DOI}"
OUTPUT_DIR="${5:-results}"

echo "================================================================================"
echo "ANALYZING RANKING PERFORMANCE"
echo "================================================================================"
echo "Predictions file: $PREDICTIONS_FILE"
echo "Positives file: $POSITIVES_FILE"
echo "Negatives file: $NEGATIVES_FILE"
echo "Matching by: $MATCH_FIELD"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Analyze positives
echo "================================================================================"
echo "ANALYZING POSITIVES"
echo "================================================================================"
uv run python analyze_ranking_performance.py \
    --labeled "$POSITIVES_FILE" \
    --predictions "$PREDICTIONS_FILE" \
    --match-by "$MATCH_FIELD" \
    --label positive \
    --output "$OUTPUT_DIR/positives_ranking.csv"

echo ""
echo ""

# Analyze negatives
echo "================================================================================"
echo "ANALYZING NEGATIVES"
echo "================================================================================"
uv run python analyze_ranking_performance.py \
    --labeled "$NEGATIVES_FILE" \
    --predictions "$PREDICTIONS_FILE" \
    --match-by "$MATCH_FIELD" \
    --label negative \
    --output "$OUTPUT_DIR/negatives_ranking.csv"

echo ""
echo ""
echo "================================================================================"
echo "ANALYSIS COMPLETE"
echo "================================================================================"
echo "Results saved to:"
echo "  - $OUTPUT_DIR/positives_ranking.csv"
echo "  - $OUTPUT_DIR/negatives_ranking.csv"
