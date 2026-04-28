# Real Estate Market Analysis

End-to-end machine learning project analyzing Taiwan's real estate market.
Predicts property prices, classifies price tiers, segments locations,
and forecasts market trends — served via a production REST API.

---

## Dataset
**Taiwan Real Estate Dataset** — 414 properties, New Taipei City (2012–2013)  
Source: [Kaggle](https://www.kaggle.com/datasets/quantbruce/real-estate-price-prediction)

Features: transaction date, house age, MRT distance,
convenience stores, GPS coordinates, price per unit area.

---

## Quick Start

### 1. Clone
```bash
git clone https://github.com/yourusername/real-estate-market-analysis.git
cd real-estate-market-analysis
```

### 2. Install
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline
```bash
python scripts/run_pipeline.py
```

### 4. Train Models
```bash
python scripts/train_models.py
```

### 5. Start API
```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

### 6. Run Tests
```bash
pytest tests/ -v
```

### 7. Docker
```bash
docker-compose -f docker/docker-compose.yml up
```

---

## Project Structure
├── data/               # Raw + processed data
├── src/
│   ├── data/           # Loader + preprocessor
│   ├── models/         # 4 ML models
│   ├── explainability/ # SHAP explainer
│   ├── analysis/       # Insights + statistical tests
│   ├── visualization/  # Chart generation
│   └── api/            # FastAPI app
├── models/             # Saved model files
├── reports/            # CSVs + figures
├── tests/              # 35+ unit tests
├── scripts/            # Pipeline + training scripts
└── docker/             # Dockerfiles + compose

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Health check + model status |
| GET | `/statistics` | Dataset summary statistics |
| POST | `/predict/price` | Predict price per unit area |
| POST | `/classify/price` | Classify as Low/Medium/High |
| POST | `/predict/segment` | Predict location segment |
| GET | `/forecast` | Revenue forecast (N months) |

API docs available at `http://localhost:8000/docs`

---

## Model Performance

| Model | Metric | Score |
|---|---|---|
| Price Predictor (GradientBoosting) | R² | 0.77 |
| Price Predictor | CV R² | 0.60 ± 0.13 |
| Price Classifier | Accuracy | 76% |
| Price Classifier | CV Accuracy | 73% ± 3% |
| Location Segmentation (KMeans) | Silhouette | 0.36 |
| Time Series Forecast | R² | 0.73 |

---

## Key Findings
- **MRT proximity is the #1 price driver** — properties within 500m cost 57% more
- **Convenience stores strongly correlate with price** — r=0.57, p<0.001
- **New homes command 45% premium** over middle-aged properties
- **4 distinct market segments** identified by location + price + amenities

See [RESULTS.md](RESULTS.md) for full analysis.

---

## Technologies

| Category | Tools |
|---|---|
| Data | pandas, numpy |
| ML | scikit-learn, scipy |
| Explainability | SHAP |
| API | FastAPI, uvicorn, pydantic |
| Testing | pytest, httpx |
| Visualization | matplotlib |
| Deployment | Docker, GitHub Actions |

---

## License
MIT