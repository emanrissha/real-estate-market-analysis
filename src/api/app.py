"""
FastAPI main application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.endpoints import router, load_models

app = FastAPI(
    title="Real Estate Market Analysis API",
    description="""
## Taiwan Real Estate Market Analysis

This API provides machine learning predictions and insights
based on the Taiwan Real Estate dataset (414 properties).

### Models
- **Price Predictor** — GradientBoosting, R² = 0.77
- **Price Classifier** — Low / Medium / High tier, Accuracy = 76%
- **Location Segmentation** — 4 KMeans clusters, Silhouette = 0.36
- **Time Series Forecast** — Monthly avg price trend, R² = 0.73

### Key Findings
- MRT distance is the strongest price driver
- Properties within 500m of MRT cost 57% more on average
- Convenience stores show strong positive correlation (r=0.57)
- New homes command a 45% premium over middle-aged homes
    """,
    version="1.0.0",
    contact={
        "name": "Real Estate Analysis",
        "url": "https://github.com/yourusername/real-estate-market-analysis"
    },
    license_info={"name": "MIT"}
)

# CORS — allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load all models at startup
@app.on_event("startup")
async def startup_event():
    print("🚀 Loading models...")
    load_models()
    print("✅ All models loaded")

# Include routes
app.include_router(router)