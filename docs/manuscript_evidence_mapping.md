# Manuscript Claim-to-Evidence Mapping

This table maps manuscript claims to downloadable evidence files in the repository.

## Performance Claims

| Claim | Evidence File | Download URL | Verification Command |
|---|---|---|---|
| Runtime per compound is in millisecond range | `Independent Reproducibility Validation/validation_2026-02-06_Benjamin_Hou/independent_validation_results/validation_report.json` | https://raw.githubusercontent.com/Benjamin-JHou/ChemWeaver/main/Independent%20Reproducibility%20Validation/validation_2026-02-06_Benjamin_Hou/independent_validation_results/validation_report.json | `python3 - <<'PY'\nimport json\np='Independent Reproducibility Validation/validation_2026-02-06_Benjamin_Hou/independent_validation_results/validation_report.json'\nobj=json.load(open(p))\nprint(obj['results']['runtime_per_compound_ms'])\nPY` |
| Figure-level runtime comparison | `Independent Reproducibility Validation/validation_2026-02-06_Benjamin_Hou/figure_6_output/validation_data.csv` | https://raw.githubusercontent.com/Benjamin-JHou/ChemWeaver/main/Independent%20Reproducibility%20Validation/validation_2026-02-06_Benjamin_Hou/figure_6_output/validation_data.csv | `cat 'Independent Reproducibility Validation/validation_2026-02-06_Benjamin_Hou/figure_6_output/validation_data.csv'` |

## Uncertainty Claims

| Claim | Evidence File | Download URL | Verification Command |
|---|---|---|---|
| Uncertainty statistics are calibrated and bounded | `Independent Reproducibility Validation/validation_2026-02-06_Benjamin_Hou/independent_validation_results/validation_report.json` | https://raw.githubusercontent.com/Benjamin-JHou/ChemWeaver/main/Independent%20Reproducibility%20Validation/validation_2026-02-06_Benjamin_Hou/independent_validation_results/validation_report.json | `python3 - <<'PY'\nimport json\np='Independent Reproducibility Validation/validation_2026-02-06_Benjamin_Hou/independent_validation_results/validation_report.json'\nobj=json.load(open(p))\nprint(obj['results']['uncertainty_statistics'])\nPY` |
| Figure-level uncertainty panel and metadata | `Independent Reproducibility Validation/validation_2026-02-06_Benjamin_Hou/figure_6_output/figure_6_supplementary.json` | https://raw.githubusercontent.com/Benjamin-JHou/ChemWeaver/main/Independent%20Reproducibility%20Validation/validation_2026-02-06_Benjamin_Hou/figure_6_output/figure_6_supplementary.json | `cat 'Independent Reproducibility Validation/validation_2026-02-06_Benjamin_Hou/figure_6_output/figure_6_supplementary.json'` |

## Reproducibility Claims

| Claim | Evidence File | Download URL | Verification Command |
|---|---|---|---|
| Independent reproducibility pass (overall) | `Independent Reproducibility Validation/INDEPENDENT_REPRODUCIBILITY_REPORT.md` | https://raw.githubusercontent.com/Benjamin-JHou/ChemWeaver/main/Independent%20Reproducibility%20Validation/INDEPENDENT_REPRODUCIBILITY_REPORT.md | `cat 'Independent Reproducibility Validation/INDEPENDENT_REPRODUCIBILITY_REPORT.md'` |
| Raw independent validation outputs | `Independent Reproducibility Validation/validation_2026-02-06_Benjamin_Hou/independent_validation_results/figure_6_data.json` | https://raw.githubusercontent.com/Benjamin-JHou/ChemWeaver/main/Independent%20Reproducibility%20Validation/validation_2026-02-06_Benjamin_Hou/independent_validation_results/figure_6_data.json | `cat 'Independent Reproducibility Validation/validation_2026-02-06_Benjamin_Hou/independent_validation_results/figure_6_data.json'` |
| Reproduction entrypoint script | `scripts/reproduce_figure6.sh` | https://raw.githubusercontent.com/Benjamin-JHou/ChemWeaver/main/scripts/reproduce_figure6.sh | `bash scripts/reproduce_figure6.sh` |

## Figure Artifacts

- `Independent Reproducibility Validation/validation_2026-02-06_Benjamin_Hou/figure_6_output/figure_6.png`
- `Independent Reproducibility Validation/validation_2026-02-06_Benjamin_Hou/figure_6_output/figure_6.pdf`
- `Independent Reproducibility Validation/validation_2026-02-06_Benjamin_Hou/figure_6_output/figure_6.eps`
