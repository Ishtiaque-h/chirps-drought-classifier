# Methods, Sources, and Evidence Index

This file lists the source-cited methods and result artifacts that should be
carried into the manuscript. It is intentionally compact: use it as a checklist
while writing Methods, Results, and figure captions.

## Core Data and Index Methods

| Component | Manuscript Use | Source/Citation |
|---|---|---|
| CHIRPS v3 monthly precipitation | Primary gridded precipitation record for SPI and drought labels | Funk et al. (2026), `https://doi.org/10.1038/s41597-026-07096-4`; Funk et al. (2015), `https://doi.org/10.1038/sdata.2015.66` |
| Standardized Precipitation Index | SPI-1/3/6/12 drought index calculation and dry/normal/wet thresholding | WMO SPI User Guide, WMO-No. 1090, `https://library.wmo.int/idurl/4/39629` |
| Brier Score / BSS / decomposition | Probability-skill evaluation against climatology | Murphy (1973), `https://ui.adsabs.harvard.edu/abs/1973JApMe..12..595M/abstract` |
| PRISM monthly precipitation | Independent Central Valley precipitation-product validation | PRISM Climate Group, `https://prism.oregonstate.edu/?id=US`; Daly et al. (2008), `https://doi.org/10.1002/joc.1688` |
| NMME/CFSv2 benchmark context | Operational/seasonal precipitation and land-surface forecast benchmark context | CPC NMME data access, `https://www.cpc.ncep.noaa.gov/products/NMME/data.html`; NCEI NMME overview, `https://www.ncei.noaa.gov/products/weather-climate-models/north-american-multi-model`; CPC probability NetCDF archive, `https://ftp.cpc.ncep.noaa.gov/NMME/prob/netcdf/`; NCEI CFSv2 THREDDS precipitation catalog, `https://www.ncei.noaa.gov/thredds/catalog/model-nmme_cfs_v2_pr_6h_agg/files/catalog.html`; NCEI CFSv2 monthly means catalog, `https://www.ncei.noaa.gov/thredds/catalog/model-cfs_v2_for_mm/catalog.html`; Kirtman et al. (2014), `https://doi.org/10.1175/BAMS-D-12-00050.1` |
| SubX future benchmark context | Future subseasonal forecast benchmark path | Pegion et al. (2019), `https://doi.org/10.1175/BAMS-D-18-0270.1` |
| SPI-12 regionalization analogy | Regionalization/teleconnection diagnostic framing | Molosiwa et al. (2026), `https://doi.org/10.1007/s00704-026-06154-6` |

## Regional Mask Sources

These are already compiled in `table03_mask_methods.csv`; cite them in the data
and study-area methods section.

| Region | Mask Type | Source |
|---|---|---|
| Central Valley | DWR Bulletin 118 groundwater basin polygons | `https://gis.water.ca.gov/arcgis/rest/services/Geoscientific/i08_B118_CA_GroundwaterBasins/FeatureServer/0/query` |
| Southern Great Plains | EPA Level III ecoregions, South Central Semi-Arid Prairies | `https://dmap-prod-oms-edc.s3.us-east-1.amazonaws.com/ORD/Ecoregions/us/us_eco_l3_state_boundaries.zip` |
| Murray-Darling | Murray-Darling Basin Boundary - Water Act 2007 | `https://data.gov.au/geoserver/murray-darling-basin-boundary/wfs?request=GetFeature&typeName=ckan_4ede9aed_5620_47db_a72b_0b3aa0a3ced0&outputFormat=json` |
| Mediterranean Spain | MITECO terrestrial river basin districts | `https://wmts.mapama.gob.es/sig-api/ogc/features/v1/collections/agua%3ADemarcaciones_ET/items` |
| Horn of Africa | Natural Earth country-intersection mask | `https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson` |

Horn caveat for manuscript text:

> The Horn of Africa run uses a country-intersection land mask over Djibouti,
> Eritrea, Ethiopia, Kenya, and Somalia. It is useful for regional
> generalization but should not be described as a hydrologic basin,
> livelihood-zone, or agroecological mask.

## Paper Evidence Pack Files

| Artifact | Purpose |
|---|---|
| `table01_master_evidence.csv` | Full consolidated evidence table across canonical, seasonal, temporal, PRISM, multiregion, feature-extension, EDL, and operational checks |
| `table02_headline_results.csv` | Compact manuscript headline table |
| `table03_mask_methods.csv` | Source-cited mask methods and retained-cell fractions |
| `table04_temporal_robustness.csv` | Rolling holdout control for test-period representativeness |
| `table05_seasonal_signal_audit.csv` | Seasonal BSS interpreted alongside event-tracking diagnostics |
| `table06_regionalization_mechanism.csv` | SPI-12 teleconnection/regionalization mechanism summary |
| `table07_evaluation_inflation_audit.csv` | Invalid-protocol audit quantifying skill inflation from random row splits, pixel-level inference, and overlapping SPI targets |
| `table08_transition_target_summary.csv` | Eligible-pixel drought-onset/termination transition diagnostic and replication checks |
| `table09_landsurface_added_value.csv` | CFSv2 root-zone soil-moisture added-value diagnostic against same-target ERA5-Land persistence |
| `operational_nmme_cpc_spi3_lead3_monthly_scores.csv` | Forecast-informed CPC NMME anomaly benchmark for Central Valley SPI-3 lead-3 |
| `operational_nmme_cpc_spi6_lead6_monthly_scores.csv` | Forecast-informed CPC NMME anomaly benchmark for Central Valley SPI-6 lead-6 |
| `operational_nmme_cpc_prob_spi1_lead1_monthly_scores.csv` | Official CPC NMME below-normal probability benchmark for Central Valley SPI-1 lead-1 |
| `operational_nmme_cpc_prob_spi3_lead3_monthly_scores.csv` | Official CPC NMME below-normal probability benchmark for Central Valley SPI-3 lead-3 |
| `operational_nmme_cpc_prob_spi6_lead6_monthly_scores.csv` | Official CPC NMME below-normal probability benchmark for Central Valley SPI-6 lead-6 |
| `ncei_cfsv2_cvalley_spi3_lead3_forecast.csv` | NOAA NCEI THREDDS CFSv2 individual-run precipitation accumulated over the Central Valley SPI-3 lead-3 target window |
| `ncei_cfsv2_cvalley_spi3_lead3_forecast_rawamount.csv` | Same NCEI CFSv2 extraction scored as raw accumulated precipitation rather than forecast anomaly |
| `operational_ncei_cfsv2_spi3_lead3_monthly_scores.csv` | NCEI CFSv2 SPI-3 lead-3 anomaly benchmark scored with validation-only calibration |
| `operational_ncei_cfsv2_spi3_lead3_rawamount_monthly_scores.csv` | NCEI CFSv2 SPI-3 lead-3 raw accumulated precipitation benchmark scored with validation-only calibration |
| `landsurface_cfsv2_rzsm_lead1_forecast.csv` | NCEI CFSv2 monthly-mean `flxf` soil-water forecast extraction for the Central Valley root-zone soil-moisture target |
| `landsurface_cfsv2_rzsm_lead1_monthly_scores.csv` | Forecast-informed CFSv2 root-zone soil-moisture benchmark scored against ERA5-Land root-zone dry-fraction observations |
| `landsurface_cfsv2_rzsm_lead1_skipped_months.csv` | Coverage audit for missing NCEI CFSv2 monthly-mean catalogs/files |
| `landsurface_cfsv2_rzsm_lead1_allcycles_monthly_scores.csv` | Strict four-cycle Central Valley CFSv2 root-zone soil-moisture replication |
| `landsurface_cfsv2_rzsm_southern_great_plains_lead1_allcycles_monthly_scores.csv` | Southern Great Plains four-cycle CFSv2 root-zone soil-moisture replication |
| `landsurface_cfsv2_rzsm_mediterranean_spain_lead1_allcycles_monthly_scores.csv` | Mediterranean Spain four-cycle CFSv2 root-zone soil-moisture replication |
| `landsurface_added_value_diagnostics.csv` | Paired Brier-score comparison of CFSv2 RZSM forecasts against raw persistence on matched months |
| `transition_target_summary.csv` | Eligible-pixel onset/termination summary showing whether transition targets replicate across geometry and regions |
| `memory_target_spi6_lead6_summary.csv` | Central Valley SPI-6 lead-6 memory-target checkpoint comparing lag/climate XGBoost, ERA5-Land soil-memory XGBoost, persistence, and external CPC NMME rows |
| `memory_target_spi6_lead6_nmme_coverage_audit.csv` | Audit showing CPC NMME real-time coverage has zero overlap with the 1991-2016 ML training period |
| `fig01_headline_bss_forest.png` | Forest plot of key model/feature/seasonal BSS checkpoints |
| `fig02_multiregion_bss_forest.png` | Multi-region selected BSS forest plot |
| `fig03_seasonal_bss_vs_tracking.png` | Seasonal BSS vs calibrated event-tracking correlation |
| `fig04_temporal_holdout_bss.png` | Rolling holdout BSS forest plot |
| `fig05_mask_retention.png` | Retained-cell fractions for regional masks |
| `manuscript_methods_draft.md` | Source-cited Methods draft tied to the current evidence tables |
| `manuscript_claims_audit.md` | Allowed-claims and overclaim-risk checklist for manuscript writing |

## Claim-to-Evidence Map

| Manuscript Claim | Required Evidence |
|---|---|
| Best Central Valley calibrated checkpoint is tied with climatology | `table02_headline_results.csv`, XGB-Spatial row; `fig01_headline_bss_forest.png` |
| Added met/soil/atmospheric/EDL paths do not solve the problem | `table02_headline_results.csv`, feature-extension and EDL rows |
| Forecast-informed NMME/CFSv2 products provide detectable but non-robust operational signal | `table02_headline_results.csv`, operational anomaly, probability, and CFSv2 SPI-3 rows |
| Forecast-informed CFSv2 root-zone soil moisture supports target reframing but not broad added value | `table02_headline_results.csv`, Central Valley all-cycle row, Southern Great Plains persistence rows, Mediterranean Spain all-cycle row, `table09_landsurface_added_value.csv`, and skipped-month audits |
| Memory-target and soil-memory checkpoint does not establish event-tracking skill | `table02_headline_results.csv`, `memory_target_spi6_lead6_summary.csv`, `memory_target_spi6_lead6_nmme_coverage_audit.csv` |
| Transition targets are diagnostic, not a robust positive SPI claim | `table08_transition_target_summary.csv`, `transition_target_summary.csv` |
| Less rigorous evaluation protocols can manufacture apparent skill | `table07_evaluation_inflation_audit.csv`; `evaluation_inflation_audit_summary.txt` |
| 2021-2026 is not the only weak evaluation window | `table04_temporal_robustness.csv`; `fig04_temporal_holdout_bss.png` |
| Multi-region tests do not reveal broad robust positive SPI-1 skill | `table02_headline_results.csv`, multi-region rows; `fig02_multiregion_bss_forest.png` |
| Regional geometry matters and must be source-cited | `table03_mask_methods.csv`; `fig05_mask_retention.png` |
| Seasonal SPI-3/SPI-6 does not broadly fix skill | `table05_seasonal_signal_audit.csv`; `fig03_seasonal_bss_vs_tracking.png` |
| Mediterranean Spain SPI-6 robust-positive result is a calibration-shift exception | `table05_seasonal_signal_audit.csv`, Mediterranean Spain SPI-6 lead-6 Niño3.4 row |
| Regionalization shows mechanism signal but not reliable forecast conversion | `table06_regionalization_mechanism.csv` |

## Terms to Use Carefully

- Use **"positive point estimate"** only when the CI crosses zero.
- Use **"robust positive"** only when the lower CI bound is above zero.
- Use **"tied with climatology"** for small BSS values with intervals crossing zero.
- Use **"calibration-shift exception"** for the Mediterranean Spain SPI-6 robust-positive row unless a follow-up proves temporal event tracking.
- Use **"forecast-informed land-surface target"** for the CFSv2 RZSM result; do not call it SPI precipitation skill or added value over persistence from the current monthly-mean extraction.
- Use **"transition-target diagnostic"** for onset/termination results; do not present the Central Valley rectangular onset row as a replicated positive forecast claim.
- Use **"mechanism evidence"** for SHAP/regionalization; do not call it causal attribution.
