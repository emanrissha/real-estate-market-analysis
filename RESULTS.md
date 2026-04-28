# Results & Findings

## Dataset
- **Source:** Taiwan Real Estate Market Dataset (Kaggle)
- **Size:** 414 properties
- **Period:** August 2012 — July 2013
- **Location:** New Taipei City, Taiwan

---

## Key Findings

### 1. MRT Distance is the Strongest Price Driver
Properties within 500m of an MRT station cost **57% more** on average than those further away.

| MRT Category | Distance | Avg Price/Unit | Count |
|---|---|---|---|
| Very Close | < 500m | 46.24 | 211 |
| Close | 500–1500m | 34.72 | 104 |
| Far | 1500–5000m | 24.28 | 94 |
| Very Far | > 5000m | 14.92 | 5 |

**T-Test result:** t=16.02, p<0.001 ✅ Highly significant

---

### 2. Convenience Stores Show Strong Positive Correlation
The number of nearby convenience stores is strongly correlated with price.

- **Pearson r = 0.57** (strong positive)
- **p < 0.001** ✅ Highly significant
- Properties near 9–10 stores average **51.73** vs **26.46** near zero stores

---

### 3. House Age Has a Significant Negative Effect
Newer homes command a significant price premium.

| Age Category | Avg Price/Unit |
|---|---|
| New (0–5 yrs) | 47.29 |
| Recent (5–15 yrs) | 38.78 |
| Middle-aged (15–30 yrs) | 32.64 |
| Old (30+ yrs) | 37.81 |

- **Pearson r = -0.21**, p<0.001 ✅ Significant
- **ANOVA F=18.88**, p<0.001 ✅ Significant difference across groups
- Note: Old homes recover slightly — likely due to location in established areas

---

### 4. Location Segments Tell a Clear Story
KMeans clustering (4 segments, Silhouette=0.36) reveals distinct market tiers:

| Segment | Label | Count | Avg Price | Avg MRT | Avg Stores |
|---|---|---|---|---|---|
| 0 | Suburban | 97 | 25.42 | 1670m | 2.1 |
| 1 | Suburban-Mid | 80 | 38.49 | 789m | 1.9 |
| 2 | Premium | 201 | 47.33 | 318m | 6.6 |
| 3 | Remote | 36 | 18.46 | 4435m | 0.2 |

---

## Model Performance

| Model | Metric | Score |
|---|---|---|
| Price Predictor | R² (test) | 0.7661 |
| Price Predictor | CV R² | 0.5953 ± 0.13 |
| Price Predictor | MAE | 4.31 per unit |
| Price Classifier | Accuracy | 76.0% |
| Price Classifier | CV Accuracy | 73.2% ± 3% |
| Price Classifier | Low F1 | 0.84 |
| Location Segmentation | Silhouette | 0.3591 |
| Time Series Forecast | R² | 0.7338 |

---

## SHAP Explainability
Top 3 features by mean absolute SHAP value:

| Feature | Mean |SHAP| |
|---|---|
| log_mrt_distance | 3.90 |
| mrt_distance | 3.77 |
| house_age | 3.35 |

MRT distance alone explains the majority of price variance.

---

## Conclusions
1. **Buy near MRT** — the single most impactful factor on price
2. **New construction commands premium** — 45% above middle-aged homes
3. **Urban amenities matter** — convenience store density is a strong proxy for urban density and price
4. **Remote properties are significantly discounted** — Segment 3 averages 18.46 vs 47.33 for Premium