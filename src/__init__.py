"""
AUTOML-Driven Predictive Analysis for Insights from Analytic Survey Data
=========================================================================

A semi-automated Machine Learning platform that streamlines the early stages of
the ML lifecycle:

    * Module 1 - Data ingestion, data preprocessing and automated EDA
    * Module 2 - Model training, evaluation and best-model selection
    * Module 3 - Model comparison (GridSearchCV based)
    * Multi-modal support - image feature extraction to complement tabular data

Reference: "AUTOML-DRIVEN PREDICTIVE ANALYSIS FOR INSIGHTS FROM ANALYTIC
SURVEY DATA", B.E. Project report, Dept. of Computer Technology,
Anna University (MIT Campus), May 2025 - Team 46.
"""

__version__ = "1.0.0"
__authors__ = ["Ramya S (2021503545)", "Rupesh A (2021503549)", "Akilan K (2021503719)"]

__all__ = [
    "config",
    "utils",
    "data_ingestion",
    "preprocessing",
    "eda",
    "model_training",
    "model_comparison",
    "image_features",
    "predict",
]
