# ChemWeaver SCI Readiness Checklist

Date: 2026-02-16  
Workspace: `ChemWeaver/`  
Remote checked: `https://github.com/Benjamin-JHou/ChemWeaver` (`main`)

---

## 1. Repository Hygiene

- [x] Removed tracked runtime cache artifacts (`__pycache__/`, `*.pyc`)
- [x] Added `.gitignore` rules for cache/environment/system files
- [x] Removed local virtual environment directory `chemweaver_test_env/`
- [x] Cleared tokenized remote URL from sibling repo `chemweaver-release/.git/config`

## 2. Documentation Consistency

- [x] Added software DOI to `CITATION.cff`
- [x] Updated README DOI badge and citation DOI
- [x] Removed broken `skill.md` / `CONTRIBUTING.md` references from README
- [x] Added local logo file at `docs/Logo.png` and linked from README
- [x] Replaced citation/documentation placeholders (`xxxxx`, `10.xxxx`, `TBD`, `[To be added]`) in:
  - `README.md`
  - `docs/citation_usage.md`
  - `docs/benchmark_philosophy.md`
  - `docs/reproducibility.md`
  - benchmark markdown drafts in `src/chemweaver/benchmark/`

## 3. Core Reproducibility Assets

- [x] Added `Independent Reproducibility Validation/` into `ChemWeaver/`
- [x] Added `PROJECT_STATUS.md` into `ChemWeaver/`
- [x] Verified local existence of core files:
  - `data/example_compounds.smi`
  - `Independent Reproducibility Validation/.../figure_6_output/figure_6.png`
  - `docs/Logo.png`
  - `CITATION.cff`
  - `README.md`

## 4. Minimal Automated Tests

- [x] Added standard-library runnable unit tests:
  - `tests/test_pipeline_smoke.py`
  - `tests/test_workflow_runner.py`
- [x] Verification command passed:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Result: `Ran 2 tests ... OK`

## 5. Smoke Validation

- [x] End-to-end workflow smoke run passed:

```bash
PYTHONPATH=src python3 -m chemweaver.workflow_runner \
  --input data/example_compounds.smi \
  --output /tmp/chemweaver_smoke_results.json \
  --confidence 0.0 --uncertainty 1.0 --top-n 3
```

- [x] Output JSON generated and parsed successfully

## 6. Remote Upload Status

- [x] Remote `main` was synchronized to `ce522ffc9d979188d54b5da92d95050df59b964b`
- [x] Cleanup/test update commits were successfully pushed
- [x] Repository sync status verified via `git ls-remote`

---

## Final Gate

Current status: **Baseline cleanup set is synced to GitHub main.**  
Note: Subsequent updates should be re-verified and pushed after local checks pass.
