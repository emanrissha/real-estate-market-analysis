"""
Tests for SHAP explainability
"""

import pytest
import pandas as pd
import os
from src.data.loader import load_raw
from src.data.preprocessor import preprocess
from src.models.price_predictor import PricePredictor
from src.explainability.shap_explainer import ShapExplainer


@pytest.fixture(scope='module')
def setup():
    df = preprocess(load_raw())
    model = PricePredictor()
    model.train(df)
    features = model.features_used
    X = df[features].fillna(df[features].mean())
    explainer = ShapExplainer()
    explainer.fit(model, X)
    return explainer, X


def test_shap_explainer_fits(setup):
    explainer, _ = setup
    assert explainer.explainer is not None


def test_shap_values_shape(setup):
    explainer, X = setup
    assert explainer.shap_values.shape == X.shape


def test_top_features_returns_dict(setup):
    explainer, _ = setup
    result = explainer.get_top_features(n=3)
    assert isinstance(result, dict)
    assert len(result) == 3


def test_top_features_mrt_is_important(setup):
    explainer, _ = setup
    top = explainer.get_top_features(n=3)
    top_names = list(top.keys())
    assert any('mrt' in name for name in top_names)


def test_bar_plot_saves_file(setup, tmp_path):
    explainer, X = setup
    save_path = str(tmp_path / 'shap_bar.png')
    explainer.bar_plot(X, save_path=save_path)
    assert os.path.exists(save_path)


def test_summary_plot_saves_file(setup, tmp_path):
    explainer, X = setup
    save_path = str(tmp_path / 'shap_summary.png')
    explainer.summary_plot(X, save_path=save_path)
    assert os.path.exists(save_path)


def test_explain_single_returns_dict(setup):
    explainer, X = setup
    result = explainer.explain_single(X.iloc[:1])
    assert isinstance(result, dict)
    assert len(result) > 0