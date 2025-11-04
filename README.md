# Central Valley Drought Classifier 

This project uses **CHIRPS** satellite-based precipitation data to analyze and classify drought risk in **California’s Central Valley**. 
The focus is to prepare a portfolio-ready, clean, and reproducible ML project.

## Goals
- Classify drought conditions in the **California’s Central Valley** using **CHIRPS** monthly precipitation data (1991–2025).
- Produce clear **maps and charts** of rainfall anomalies and drought classes.
- Build a **reproducible ML pipeline** (data ➜ features ➜ model ➜ evaluation).

## Region & Data
- **Region:** California Central Valley (≈ 35.4°N–40.6°N, 122.5°W–119.0°W).
- **Dataset:** CHIRPS v3 Monthly (0.05°), 1991–2025.
- **Why CHIRPS here?** Good statistics on blending, strong monthly skill, long record.

## Environment (with mamba)
```bash
# create env from environment.yml
mamba env create -f environment.yml
mamba activate chirps-ml
```

## Quickstart
```bash
# (1) activate env
mamba activate chirps-ml

# (2) open notebooks
jupyter lab
# run notebooks/eda.ipynb then notebooks/modeling.ipynb
```

## Project Structure
```
central-valley-drought-classifier/
├── README.md
├── environment.yml
├── data/               # (add .gitkeep) raw/ and processed/ CHIRPS files (not committed)
├── notebooks/          # EDA, modeling, evaluation
├── scripts/            # helper scripts (preprocessing, labeling)
└── outputs/            # figures, maps, metrics (not committed)
```

## 📈 Progress Log
- [x] Initialize repo, env, and README
- [x] Download CHIRPS monthly (1991–2025)
- [x] Clip to Central Valley (bbox) and save NetCDF
- [x] Compute monthly climatology (1991–2020)
- [x] Compute monthly anomalies (1991–2025)
- [x] Label drought classes (dry/normal/wet)
- [x] Exploratory data analysis (time series + spatial maps)
- [ ] Build model-ready dataset
- [ ] Baseline model + metrics
- [ ] Final visualizations and report

## Pipeline (high-level)
```
graph TD;

    A(("CHIRPS v3 Monthly (1991–2025, global, yearly .nc)"))
    G(("Modeling + Maps + Report"))
    B["Download (parallel by year)"]
    C["Clip to Central Valley bbox"]
    D["Monthly Climatology (1991–2020)"]
    E["Monthly Anomalies (1991–2025)"]
    F["Drought classes: dry/normal/wet"]

    A --> B
    B --> C
    C --> D
    C --> E
    D --> |"pr - monthly_climatology"| E
    E --> F
    F --> G
```                                   
## Key Data Artifacts

| File Path                                                      | Description                                                                      | Dimensions                        |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------- | --------------------------------- |
| `data/processed/chirps_v3_monthly_cvalley_1991_2025.nc`        | Regional subset of  v3 monthly precipitation for  (1991–2025)                    | time × lat × lon (415 × 104 × 70) |
| `data/processed/chirps_v3_monthly_cvalley_clim_1991_2020.nc`   | Long-term monthly climatology (1991–2020 baseline means)                         | month × lat × lon (12 × 104 × 70) |
| `data/processed/chirps_v3_monthly_cvalley_anom_1991_2025.nc`   | Monthly precipitation anomalies (actual − climatology)                           | time × lat × lon (415 × 104 × 70) |
| `data/processed/chirps_v3_monthly_cvalley_labels_1991_2025.nc` | Drought class labels (dry / normal / wet) with 20th / 80th percentile thresholds | time × lat × lon (415 × 104 × 70) |
| `outputs/drought_shares.csv`                                   | Monthly fraction of the region in each drought class                             | time × 3 classes                  |
| `outputs/drought_shares_stacked.png`                           | Stacked area plot of dry / normal / wet area shares over time                    | —                                 |
| `outputs/drought_map_YYYY-MM.png`                              | Spatial drought class map for selected months                                    | lat × lon                         |

## Acknowledgement
Used AI tools (ChatGpt & Gemini) to design, improve, and test code.

## References
- CHIRPS: Climate Hazards Group, UCSB — https://www.chc.ucsb.edu/data/chirps

