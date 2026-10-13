"""Cross-section of CAR on event characteristics.

Default engine is pyfixest with firm-clustered standard errors.
statsmodels with the same dummy fixed effects is the fallback.
Coefficients are associations. They are not treatment effects.
Within R² is reported for specifications that absorb fixed effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_event.config import window_suffix


def regression_covariates(frame: pd.DataFrame, requested: list[str], include_peer: bool) -> list[str]:
    columns = list(requested)
    if include_peer and "peer_count_pm5" not in columns:
        columns.append("peer_count_pm5")
    kept = []
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if int(values.notna().sum()) >= 8 and float(values.std(ddof=0) or 0) > 0:
            kept.append(column)
    return kept


def _singleton_free(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame
    for column in columns:
        counts = out[column].value_counts()
        keep_levels = counts[counts >= 2].index
        out = out[out[column].isin(keep_levels)]
    return out


def _fe_columns(frame: pd.DataFrame) -> list[list[str]]:
    """Return the FE sets that this sample can support, from none to two-way."""
    sets: list[list[str]] = [[]]
    industry = "industry" in frame.columns and frame["industry"].value_counts().ge(2).sum() >= 2
    year = "year" in frame.columns and frame["year"].value_counts().ge(2).sum() >= 2
    if industry:
        sets.append(["industry"])
    if industry and year:
        sets.append(["industry", "year"])
    elif year:
        sets.append(["year"])
    return sets


def _extract(model, title: str, fe: list[str], cluster: str, engine: str, n_input: int) -> dict:
    tidy = model.tidy()
    raw_g = getattr(model, "_G", None)
    clusters = None
    if isinstance(raw_g, (list, tuple)) and raw_g:
        clusters = int(raw_g[0])
    else:
        try:
            clusters = int(raw_g)
        except (TypeError, ValueError):
            clusters = None
    has_fe = bool(getattr(model, "_has_fixef", False)) or bool(fe)
    r2_within = getattr(model, "_r2_within", None)
    if not has_fe:
        r2_within = None
    return {
        "title": title,
        "terms": [str(name) for name in tidy.index],
        "b": {str(k): float(v) for k, v in tidy["Estimate"].items()},
        "se": {str(k): float(v) for k, v in tidy["Std. Error"].items()},
        "p": {str(k): float(v) for k, v in tidy["Pr(>|t|)"].items()},
        "n": int(getattr(model, "_N", n_input)),
        "r2": float(model._r2) if getattr(model, "_r2", None) is not None else None,
        "r2_within": None if r2_within is None or not has_fe else float(r2_within),
        "fe_industry": "industry" in fe,
        "fe_year": "year" in fe,
        "cluster": cluster,
        "g": clusters,
        "engine": engine,
    }


def _fit_pyfixest(formula: str, data: pd.DataFrame, cluster: str):
    import pyfixest as pf

    groups = int(data[cluster].nunique())
    if groups >= 2:
        try:
            return pf.feols(formula, data=data, vcov={"CRV1": cluster})
        except Exception:
            pass
    return pf.feols(formula, data=data, vcov="HC1")


def _fit_statsmodels(formula: str, data: pd.DataFrame, cluster: str):
    import statsmodels.formula.api as smf

    groups = int(data[cluster].nunique())
    if groups >= 2:
        try:
            return smf.ols(formula, data=data).fit(
                cov_type="cluster",
                cov_kwds={"groups": data[cluster]},
            )
        except Exception:
            pass
    return smf.ols(formula, data=data).fit(cov_type="HC1")


def _from_statsmodels(result, title: str, fe: list[str], cluster: str, n_input: int) -> dict:
    names = [str(name) for name in result.params.index if not str(name).startswith("C(")]
    b, se, p = {}, {}, {}
    for name in names:
        b[name] = float(result.params[name])
        se[name] = float(result.bse[name])
        p[name] = float(result.pvalues[name])
    return {
        "title": title,
        "terms": names,
        "b": b,
        "se": se,
        "p": p,
        "n": int(result.nobs),
        "r2": float(result.rsquared),
        "r2_within": None,
        "fe_industry": "industry" in fe,
        "fe_year": "year" in fe,
        "cluster": cluster,
        "g": int(n_input),
        "engine": "statsmodels",
    }


def fit_car_regressions(
    events: pd.DataFrame,
    *,
    regression_window: tuple[int, int],
    covariates: list[str],
    cluster: str,
    return_scale: float,
) -> list[dict]:
    suffix = window_suffix(*regression_window)
    outcome = f"car_{suffix}"
    if events.empty or outcome not in events.columns:
        return []
    columns = ["event_id", "firm_id", "industry", "year", outcome, *covariates]
    columns = [col for col in columns if col in events.columns]
    work = events[columns].copy()
    work["car"] = pd.to_numeric(work[outcome], errors="coerce") * return_scale
    for column in covariates:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["car", *covariates]).reset_index(drop=True)
    if len(work) < 8:
        return []
    if cluster not in work.columns:
        cluster = "firm_id"
    work[cluster] = work[cluster].astype(str)
    pieces = ["1"]
    if covariates:
        pieces.append(" + ".join(covariates))
    formulas = []
    seen = set()
    for fe in _fe_columns(work):
        for rhs in pieces:
            if rhs != "1" and not covariates:
                continue
            if rhs == "1" and fe:
                continue
            key = (rhs, tuple(fe))
            if key in seen:
                continue
            seen.add(key)
            formulas.append((rhs, fe))
    models = []
    for index, (rhs, fe) in enumerate(formulas, start=1):
        sample = _singleton_free(work, fe) if fe else work
        if len(sample) < 8:
            continue
        formula = "car ~ " + rhs
        if fe:
            formula += " | " + " + ".join(fe)
        title = f"({index})"
        try:
            fitted = _fit_pyfixest(formula, sample, cluster)
            models.append(_extract(fitted, title, fe, cluster, "pyfixest", len(sample)))
            continue
        except Exception:
            sm_formula = "car ~ " + rhs
            if fe:
                sm_formula += " + " + " + ".join(f"C({col})" for col in fe)
            fitted = _fit_statsmodels(sm_formula, sample, cluster)
            models.append(_from_statsmodels(fitted, title, fe, cluster, sample[cluster].nunique()))
    return models
