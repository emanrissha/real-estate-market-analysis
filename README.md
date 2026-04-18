# 🏠 Real Estate Market Analysis

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📊 Project Overview

An end-to-end **advanced real estate market analysis** project that processes property transactions and customer satisfaction data. The project includes data cleaning, exploratory analysis, machine learning models, REST API, interactive dashboard, and production-ready deployment.

### Key Features
- ✅ Data cleaning and preprocessing (pandas, numpy)
- ✅ Descriptive statistics and aggregations
- ✅ Age and price correlation analysis
- ✅ **4 Machine Learning Models** (Price Prediction, Customer Segmentation, Satisfaction Classification, Time Series Forecast)
- ✅ **REST API** with FastAPI (8+ endpoints)
- ✅ **Interactive Dashboard** with Streamlit
- ✅ **Unit Tests** with pytest (30+ tests)
- ✅ **Docker** containerization
- ✅ **CI/CD** pipeline with GitHub Actions

---

## 📁 Project Structure
real-estate-market-analysis/
├── data/
│ ├── raw/ # Raw CSV files
│ └── processed/ # Cleaned and merged data
├── src/
│ ├── data/ # Data loading & cleaning
│ ├── features/ # Feature engineering
│ ├── analysis/ # Statistical analysis
│ ├── models/ # ML models (4)
│ ├── visualization/ # Charts & dashboard
│ └── api/ # FastAPI endpoints
├── tests/ # Unit tests (30+)
├── reports/ # Generated CSV & PNG
├── notebooks/ # Jupyter notebooks
├── config/ # YAML configurations
├── docker/ # Docker files
├── scripts/ # Automation scripts
└── .github/workflows/ # CI/CD pipelines

text

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/real-estate-market-analysis.git
cd real-estate-market-analysis
2. Install Dependencies
bash
pip install -r requirements/base.txt
pip install -r requirements/dev.txt
pip install -r requirements/prod.txt
3. Run the Full Pipeline
bash
python scripts/run_pipeline.py
4. Train All ML Models
bash
python scripts/train_models.py
5. Run Tests
bash
pytest tests/ -v
🔌 API Endpoints
Method	Endpoint	Description
GET	/	API information
GET	/health	Health check
POST	/predict_price	Predict property price
POST	/predict_segment	Predict customer segment
GET	/properties	Get property list
GET	/statistics	Get summary statistics
GET	/v1/analysis/by-building	Building type analysis
GET	/v1/analysis/by-country	Country analysis
Start API
bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
📊 Interactive Dashboard
bash
streamlit run src/visualization/interactive_dashboard.py
🐳 Docker Deployment
bash
# Build image
docker build -t real-estate-analysis -f docker/Dockerfile .

# Run with docker-compose
docker-compose -f docker/docker-compose.yml up
📈 Model Performance
Model	Metric	Score
Price Predictor	R² Score	0.8521
Price Predictor	MAE	$27,691
Customer Segmentation	Clusters	4 segments
Satisfaction Classifier	Accuracy	~85%
📊 Key Insights
Age-Price Correlation: -0.175 (weak negative)

Most Active Age Group: 31-36 years

Top State: California (66.3% of properties)

Top Country: USA (84.6% of properties)

🛠️ Technologies Used
Category	Technologies
Data Processing	pandas, numpy
ML & Analysis	scikit-learn, scipy, statsmodels
Visualization	matplotlib, seaborn, plotly, streamlit
API	FastAPI, uvicorn
Testing	pytest
Container	Docker, Docker Compose
CI/CD	GitHub Actions
📝 License
MIT License

👥 Author
Your Name

⭐ Star this repository if you find it useful!
