"""Variable dictionary and variance inflation for the CAR regression.

VIF flags unstable associations. It is not a rule for dropping a variable,
and it does not identify a mechanism.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ashare_event.labels import coef_label


def variance_inflation(frame: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    usable = []
    for column in covariates:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if int(values.notna().sum()) >= 8 and float(values.std(ddof=0) or 0) > 0:
            usable.append(column)
    if not usable:
        return pd.DataFrame(columns=["variable", "vif", "n"])
    data = frame[usable].apply(pd.to_numeric, errors="coerce").dropna()
    if len(usable) == 1 or len(data) < 8:
        return pd.DataFrame([{"variable": name, "vif": 1.0, "n": int(len(data))} for name in usable])
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    exog = np.column_stack([np.ones(len(data)), data.to_numpy(dtype=float)])
    rows = []
    for index, name in enumerate(usable, start=1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            value = float(variance_inflation_factor(exog, index))
        if not np.isfinite(value):
            value = float("inf")
        rows.append({"variable": name, "vif": value, "n": int(len(data))})
    return pd.DataFrame(rows)


def variable_dictionary(frame: pd.DataFrame, columns: list[str], language: str = "zh") -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        rows.append(
            {
                "variable": column,
                "label": coef_label(language, column),
                "n": int(numeric.notna().sum()) if numeric.notna().any() else int(frame[column].notna().sum()),
                "missing": int(frame[column].isna().sum()) if not numeric.notna().any() else int(numeric.isna().sum()),
                "mean": float(numeric.mean()) if numeric.notna().any() else np.nan,
                "sd": float(numeric.std(ddof=1)) if int(numeric.notna().sum()) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)
