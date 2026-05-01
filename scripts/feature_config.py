#!/usr/bin/env python
"""
Shared feature configuration for drought-forecast scripts.

Base features are always expected in dataset_forecast.parquet.
Optional exogenous climate features are used automatically when present.
"""
from __future__ import annotations

from typing import Iterable

BASE_FEATURES = [
    "spi1_lag1", "spi1_lag2", "spi1_lag3",
    "spi3_lag1", "spi6_lag1",
    "pr_lag1", "pr_lag2", "pr_lag3",
    "month_sin", "month_cos",
]

OPTIONAL_EXOG_FEATURES = [
    "nino34_lag1", "nino34_lag2",
    "pdo_lag1", "pdo_lag2",
]


def get_feature_columns(df_columns: Iterable[str]) -> list[str]:
    """Return the active feature set, adding optional exogenous features if available."""
    cols = set(df_columns)
    missing_base = [f for f in BASE_FEATURES if f not in cols]
    if missing_base:
        raise ValueError(f"Missing required base features: {missing_base}")
    return BASE_FEATURES + [f for f in OPTIONAL_EXOG_FEATURES if f in cols]
