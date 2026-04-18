"""
Real Estate Market Analysis - Advanced Package
"""

__version__ = "2.0.0"
__author__ = "Real Estate Analytics Team"
__description__ = "Advanced real estate market analysis with ML and API"

from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.data.preprocessor import Preprocessor
from src.features.age_binner import AgeBinner
from src.features.price_binner import PriceBinner
from src.analysis.descriptive_stats import DescriptiveStats
from src.analysis.correlation_analysis import CorrelationAnalysis
from src.visualization.charts import RealEstateVisualizer

__all__ = [
    'DataLoader',
    'DataCleaner', 
    'Preprocessor',
    'AgeBinner',
    'PriceBinner',
    'DescriptiveStats',
    'CorrelationAnalysis',
    'RealEstateVisualizer'
]