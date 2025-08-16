"""
Utility functions for generating dummy datasets and metrics.

This module centralises the logic for producing fake data and metrics used by
both the server and client components.  By separating the generation logic
from the API you can swap these functions out for real computations when
integrating with actual datasets and models.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np


def generate_model_info(num_models: int = 3) -> List[Dict]:
    """
    Produce a list of model metadata dictionaries with names and statistics.

    Models are given tangible names relevant to banking scenarios such as fraud
    detection or churn prediction.  Each model includes labels, feature names,
    coefficients, means and variances.
    """
    model_names = [
        "Credit Card Fraud Detection",
        "Customer Churn Prediction",
        "Loan Default Risk",
    ]
    feature_pools = [
        ["transaction_amount", "transaction_type", "merchant_category", "customer_age", "card_present"],
        ["account_age", "transaction_volume", "customer_tenure", "support_calls", "monthly_fee"],
        ["income", "loan_amount", "interest_rate", "employment_length", "credit_score"],
    ]
    models: List[Dict] = []
    for i in range(num_models):
        features = feature_pools[i % len(feature_pools)]
        num_features = len(features)
        coefficients = np.random.randn(num_features).round(3).tolist()
        feature_means = np.random.rand(num_features).round(3).tolist()
        coefficient_variance = (np.random.rand(num_features) * 0.5).round(3).tolist()
        models.append(
            {
                "id": str(uuid.uuid4()),
                "name": model_names[i % len(model_names)],
                "labels": ["no", "yes"],
                "stats": {
                    "feature_names": features,
                    "coefficients": coefficients,
                    "feature_means": feature_means,
                    "coefficient_variance": coefficient_variance,
                },
            }
        )
    return models


def generate_business_metrics() -> Dict:
    """
    Generate time-series business metrics for cost, resource utilisation and model performance.
    """
    now = datetime.utcnow()
    months = [now - timedelta(days=30 * i) for i in reversed(range(12))]
    timestamps = [m.strftime("%Y-%m-%d") for m in months]

    def random_series(name: str, baseline: float) -> Dict:
        values = (baseline + 0.1 * np.random.randn(12)).round(3).tolist()
        return {"name": name, "timestamps": timestamps, "values": values}

    return {
        "cost": random_series("Cost ($k)", 100.0),
        "resource_utilization": random_series("Resource Utilization (%)", 70.0),
        "performance": random_series("Model Accuracy (%)", 85.0),
    }


def generate_drift_metrics(num_features: int = 5) -> Dict:
    """
    Produce a drift score and per‑feature drift details.

    The drift score is a random float in [0,1]; drift is considered detected if the score > 0.7.
    """
    score = random.random()
    details = {f"feature_{i+1}": round(random.random(), 3) for i in range(num_features)}
    return {
        "drift_score": round(score, 3),
        "drift_detected": score > 0.7,
        "details": details,
    }


def calculate_metrics_from_paths(training_path: str, production_path: str) -> Dict:
    """
    Simulate the calculation of metrics from provided data paths.

    In a real environment this function would read the CSV or Parquet data from
    the given paths, compute relevant statistics such as feature means or drift
    scores and return them.  The current implementation returns random values
    to simulate this behaviour.

    Parameters
    ----------
    training_path : str
        Path to the training data (ignored for dummy calculations)
    production_path : str
        Path to the production data (ignored for dummy calculations)

    Returns
    -------
    dict
        Dictionary with keys `drift` and `stats`.
    """
    # Use the same feature set as generate_model_info for consistency
    feature_names = [
        "transaction_amount",
        "transaction_type",
        "merchant_category",
        "customer_age",
        "card_present",
    ]
    num_features = len(feature_names)
    # Dummy coefficients and means
    coefficients = np.random.randn(num_features).round(3).tolist()
    feature_means = np.random.rand(num_features).round(3).tolist()
    coefficient_variance = (np.random.rand(num_features) * 0.5).round(3).tolist()
    stats = {
        "feature_names": feature_names,
        "coefficients": coefficients,
        "feature_means": feature_means,
        "coefficient_variance": coefficient_variance,
    }
    drift = generate_drift_metrics(num_features)
    return {"drift": drift, "stats": stats}