# Documentation Map

This project now has one active documentation path. Use this map to avoid
reading stale experiment notes as current conclusions.

## Start Here

1. [`README.md`](../README.md) - project overview, reproducible pipeline, and
   current headline results.
2. [`ANALYSIS.md`](../ANALYSIS.md) - detailed research assessment and strategic
   roadmap.
3. [`final_report.md`](../final_report.md) - current narrative synthesis,
   literature-informed hypothesis assessment, and manuscript strategy.
4. [`results/paper/paper_evidence_pack.md`](../results/paper/paper_evidence_pack.md)
   - manuscript-facing evidence summary generated from current result tables.

## Manuscript Drafts

- [`results/paper/manuscript_methods_draft.md`](../results/paper/manuscript_methods_draft.md)
  - source-cited methods draft.
- [`results/paper/manuscript_results_discussion_draft.md`](../results/paper/manuscript_results_discussion_draft.md)
  - results and discussion draft.
- [`results/paper/manuscript_claims_audit.md`](../results/paper/manuscript_claims_audit.md)
  - allowed-claims and overclaim-risk checklist.
- [`results/paper/methods_sources_and_evidence_index.md`](../results/paper/methods_sources_and_evidence_index.md)
  - citation and claim-to-evidence index.

## Current Evidence Tables

The CSV files are the machine-readable source of truth. Paper table Markdown
copies are intentionally not generated to reduce documentation drift.

- [`results/report/master_results_table.csv`](../results/report/master_results_table.csv)
- [`results/report/master_results_headline.csv`](../results/report/master_results_headline.csv)
- [`results/paper/table01_master_evidence.csv`](../results/paper/table01_master_evidence.csv)
- [`results/paper/table02_headline_results.csv`](../results/paper/table02_headline_results.csv)
- [`results/paper/table03_mask_methods.csv`](../results/paper/table03_mask_methods.csv)
- [`results/paper/table04_temporal_robustness.csv`](../results/paper/table04_temporal_robustness.csv)
- [`results/paper/table05_seasonal_signal_audit.csv`](../results/paper/table05_seasonal_signal_audit.csv)
- [`results/paper/table06_regionalization_mechanism.csv`](../results/paper/table06_regionalization_mechanism.csv)

## Literature Index

- [`literature/related_works_index.csv`](../literature/related_works_index.csv)
  - tracked related works, local PDF status, and how each paper affects the
    project claim.
- [`literature/papers/`](../literature/papers/)
  - local PDFs for papers that allowed direct download.


## Rule

For new claims, update the scripts and regenerate the tables first. Do not make
paper claims from one-off Markdown notes.
