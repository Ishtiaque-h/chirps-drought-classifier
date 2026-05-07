# Documentation Map

Use this map to find the documentation acroos the project repo

## Start Here

1. [`README.md`](../README.md) - project overview, reproducible pipeline, and
   current headline results.
2. [`ANALYSIS.md`](../ANALYSIS.md) - detailed research assessment and strategic
   roadmap.
3. [`final_report.md`](../final_report.md) - current narrative synthesis,
   literature-informed hypothesis assessment, and manuscript strategy.
4. [`results/report/paper/paper_evidence_pack.md`](../results/report/paper/paper_evidence_pack.md)
   - manuscript-facing evidence summary generated from current result tables.
5. [`results/README.md`](../results/README.md) - map of manuscript-facing and
   supporting result folders.

## Manuscript Drafts

- [`results/report/paper/manuscript_methods_draft.md`](../results/report/paper/manuscript_methods_draft.md)
  - source-cited methods draft.
- [`results/report/paper/manuscript_results_discussion_draft.md`](../results/report/paper/manuscript_results_discussion_draft.md)
  - results and discussion draft.
- [`results/report/paper/manuscript_claims_audit.md`](../results/report/paper/manuscript_claims_audit.md)
  - allowed-claims and overclaim-risk checklist.
- [`results/report/paper/methods_sources_and_evidence_index.md`](../results/report/paper/methods_sources_and_evidence_index.md)
  - citation and claim-to-evidence index.

## Current Evidence Tables

The CSV files are the machine-readable source of truth. Paper table Markdown
copies are intentionally not generated to reduce documentation drift.

- [`results/report/master_results_table.csv`](../results/report/master_results_table.csv)
- [`results/report/master_results_headline.csv`](../results/report/master_results_headline.csv)
- [`results/report/paper/table01_master_evidence.csv`](../results/report/paper/table01_master_evidence.csv)
- [`results/report/paper/table02_headline_results.csv`](../results/report/paper/table02_headline_results.csv)
- [`results/report/paper/table03_mask_methods.csv`](../results/report/paper/table03_mask_methods.csv)
- [`results/report/paper/table04_temporal_robustness.csv`](../results/report/paper/table04_temporal_robustness.csv)
- [`results/report/paper/table05_seasonal_signal_audit.csv`](../results/report/paper/table05_seasonal_signal_audit.csv)
- [`results/report/paper/table06_regionalization_mechanism.csv`](../results/report/paper/table06_regionalization_mechanism.csv)
- [`results/report/paper/table07_evaluation_inflation_audit.csv`](../results/report/paper/table07_evaluation_inflation_audit.csv)
- [`results/report/paper/table08_transition_target_summary.csv`](../results/report/paper/table08_transition_target_summary.csv)
- [`results/report/paper/table09_landsurface_added_value.csv`](../results/report/paper/table09_landsurface_added_value.csv)

## Literature Index

- [`literature/related_works_index.csv`](../literature/related_works_index.csv)
  - tracked related works, local PDF status, and how each paper affects the
    project claim.
- [`literature/literature_protocol_audit.csv`](../literature/literature_protocol_audit.csv)
  - protocol-comparability audit for target, lead, predictors, validation,
    metrics, and baseline differences across related work.
- [`literature/literature_protocol_audit.md`](../literature/literature_protocol_audit.md)
  - cautious wording supported by the protocol audit.
- [`literature/papers/`](../literature/papers/)
  - local PDFs for papers that allowed direct download.


## Rule

For new claims, update the scripts and regenerate the tables first. Do not make
paper claims from one-off Markdown notes.
