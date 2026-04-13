# Branch Merge Guide

This document explains the relationship between the improvement branches and the merge strategy used to combine them.

## Branch Overview

| Branch | PR | Changes |
|--------|-----|---------|
| `copilot/evaluate-v20-implementation` | #6 | Bootstrap 95% CIs for **all** models in `evaluate_forecast_skill.py` + README rewrite (structured, with placeholder results) |
| `copilot/research-analysis-chirps-drought-classifier` | #7 | `evaluate_forecast_skill.py` improvements (CIs for core models only) + concise README rewrite (with actual results) + `ANALYSIS.md` research assessment |
| `copilot/update-readme-and-analysis` | #8 | **Combined branch** — merges the best of PR #6 and PR #7 |

## Overlap Analysis

Both branches modify the same three files relative to `v2.0`:

| File | PR #6 | PR #7 | Conflict? |
|------|-------|-------|-----------|
| `scripts/evaluate_forecast_skill.py` | Bootstrap CIs for **all** models (XGB-Spatial, ConvLSTM, LogReg, RF) | CIs only for core baselines + XGBoost; optional models show "—" | Auto-mergeable (PR #6 is a strict superset) |
| `scripts/train_forecast_xgb_spatial.py` | Validation probability saving | Identical change | No conflict |
| `README.md` | Structured rewrite with placeholder result values | Concise, insight-driven rewrite with actual result numbers | **Conflict** (3 conflict regions — both rewrote the entire file) |
| `ANALYSIS.md` | Not present | Full research assessment and strategic roadmap | No conflict (unique to PR #7) |

## What This Combined Branch Contains

This branch (`copilot/update-readme-and-analysis`) takes:

- **Code from PR #6**: Complete bootstrap 95% CIs for all models (the more rigorous version)
- **README from PR #7**: Concise, insight-driven documentation with actual results and key findings
- **ANALYSIS.md from PR #7**: Comprehensive research assessment and publication roadmap

This combination gives the default branch the most complete code AND the most informative documentation.

## Merge Procedure Into Default Branch

To merge this combined branch into `v2.0` (the default branch):

```bash
# 1. Fetch latest
git fetch origin

# 2. Checkout default branch
git checkout v2.0

# 3. Merge the combined branch
git merge origin/copilot/update-readme-and-analysis

# 4. Verify — no conflicts expected since this branch is based on v2.0
git status

# 5. Push
git push origin v2.0
```

Alternatively, merge this PR (#8) via the GitHub UI — it targets `v2.0` and should merge cleanly.

## After Merging

Once PR #8 is merged into `v2.0`, PRs #6 and #7 can be **closed without merging** — all their changes are already included.
