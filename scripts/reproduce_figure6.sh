#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VALIDATION_DIR="$ROOT_DIR/Independent Reproducibility Validation/validation_2026-02-06_Benjamin_Hou"
MPL_CACHE_DIR="${TMPDIR:-/tmp}/chemweaver-mpl-cache"

echo "Reproducing independent validation artifacts (Figure 6)..."
echo "Root: $ROOT_DIR"
echo "Validation dir: $VALIDATION_DIR"

if [[ ! -d "$VALIDATION_DIR" ]]; then
  echo "Validation directory not found: $VALIDATION_DIR" >&2
  exit 1
fi

mkdir -p "$MPL_CACHE_DIR"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$MPL_CACHE_DIR"

pushd "$VALIDATION_DIR" >/dev/null
PYTHONPATH="$ROOT_DIR/src" python3 simplified_validation.py
python3 generate_figure_6.py
popd >/dev/null

echo "Done. Generated outputs:"
echo "  $VALIDATION_DIR/figure_6_output/figure_6.png"
echo "  $VALIDATION_DIR/figure_6_output/figure_6.pdf"
echo "  $VALIDATION_DIR/figure_6_output/figure_6.eps"
echo "  $VALIDATION_DIR/independent_validation_results/validation_report.json"
