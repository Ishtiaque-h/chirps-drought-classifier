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

## 📈 Status
Project initialized ✅ — starting with EDA and anomaly labeling.

## 🔭 Roadmap
- [ ] CHIRPS monthly aggregation + anomalies
- [ ] Drought class labeling (percentiles)
- [ ] Baseline models: Logistic Regression, RandomForest, XGBoost
- [ ] Evaluation reports and confusion matrix
- [ ] Maps of drought classes (static PNGs)
- [ ] (Optional) Streamlit mini-dashboard

## 📚 References
- CHIRPS: Climate Hazards Group, UCSB — https://www.chc.ucsb.edu/data/chirps

