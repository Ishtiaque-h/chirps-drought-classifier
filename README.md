# Central Valley Drought Classifier 

This project uses **CHIRPS** satellite-based precipitation data to analyze and classify drought risk in **California’s Central Valley**. 
The focus is portfolio-ready: clean, reproducible, and easy to understand—without unnecessary complexity.

## Goals
- Classify each month as **dry / normal / wet** using CHIRPS (2015–2024).
- Produce clear **maps and charts** of rainfall anomalies and drought classes.
- Build a **reproducible ML pipeline** (data ➜ features ➜ model ➜ evaluation).

## Region & Data
- **Region:** California Central Valley (≈ 35.4°N–40.6°N, 122.5°W–119.0°W)
- **Dataset:** CHIRPS Daily (0.05°), aggregated to monthly for modeling.
- **Why CHIRPS here?** Good stati on blending, strong monthly skill, long record.

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

## Status
Project initialized ✅ — data download ✅ — Central Valley clip ✅ — next: climatology & anomalies

## 📈 Progress Log
- [x] Initialize repo, env, and README
- [x] Download CHIRPS monthly (1991–2024/2025 YTD)
- [x] Clip to Central Valley (bbox) and save NetCDF
- [ ] Compute monthly climatology (1991–2020)
- [ ] Compute monthly anomalies
- [ ] Label drought classes (dry/normal/wet)
- [ ] Baseline model + metrics
- [ ] Maps and final report assets

## Pipeline (high-level)

[ CHIRPS v3 Monthly (1991–2025, global, yearly .nc) ]
                │
                ▼
[ Download (parallel by year) ]
                │
                ▼
[ Clip to Central Valley bbox ]
                │
                ▼
[ Monthly Climatology (1991–2020) ]   [ Monthly Anomalies (1991–2025) ]
                │                                  │
                └──────────────►  (pr - monthly_climatology)  ◄──────────────┘
                                                    │
                                                    ▼
                                        [ Drought classes: dry/normal/wet ]
                                                    │
                                                    ▼
                                          [ Modeling + Maps + Report ]

## References
- CHIRPS: Climate Hazards Group, UCSB — https://www.chc.ucsb.edu/data/chirps

