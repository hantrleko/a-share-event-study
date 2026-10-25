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


_WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def _cluster_weights(rng: np.random.Generator, groups: np.ndarray) -> np.ndarray:
    uniq = pd.unique(groups)
    if len(uniq) < 12:
        draw = rng.choice(_WEBB, size=len(uniq))
    else:
        draw = rng.choice(np.array([-1.0, 1.0]), size=len(uniq))
    table = dict(zip(uniq, draw, strict=True))
    return np.array([table[group] for group in groups], dtype=float)


def wild_cluster_p(
    data: pd.DataFrame,
    covariates: list[str],
    cluster: str,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, float]:
    """Restricted wild cluster bootstrap p-values for H0: each coefficient is zero.

    These are not treatment-effect p-values. Stars in the table still use the
    cluster standard error.
    """
    if draws <= 0 or not covariates:
        return {}
    y = data["car"].to_numpy(dtype=float)
    x_cols = data[covariates].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(data)), x_cols])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    groups = data[cluster].astype(str).to_numpy()
    out = {}
    for index, name in enumerate(covariates, start=1):
        keep = [k for k in range(design.shape[1]) if k != index]
        restricted = design[:, keep]
        br, _, _, _ = np.linalg.lstsq(restricted, y, rcond=None)
        residual = y - restricted @ br
        exceed = 0
        for _ in range(draws):
            weights = _cluster_weights(rng, groups)
            y_star = restricted @ br + weights * residual
            b_star, _, _, _ = np.linalg.lstsq(design, y_star, rcond=None)
            if abs(float(b_star[index])) >= abs(float(beta[index])) - 1e-12:
                exceed += 1
        out[name] = float((1 + exceed) / (1 + draws))
    return out


def _cgm_pack(sample: pd.DataFrame, rhs: str, fe: list[str], cluster: str, title: str) -> dict | None:
    day = "event_date_aligned" if "event_date_aligned" in sample.columns else "event_date"
    if day not in sample.columns:
        return None
    work = sample.copy()
    work["_day"] = pd.to_datetime(work[day]).dt.strftime("%Y-%m-%d")
    g_firm = int(work[cluster].nunique())
    g_day = int(work["_day"].nunique())
    if g_firm < 2 or g_day < 2:
        return None
    import statsmodels.formula.api as smf
    from scipy import stats as sps

    formula = "car ~ " + rhs
    if fe:
        formula += " + " + " + ".join(f"C({col})" for col in fe)
    ols = smf.ols(formula, data=work)
    firm = ols.fit(cov_type="cluster", cov_kwds={"groups": work[cluster]})
    by_day = ols.fit(cov_type="cluster", cov_kwds={"groups": work["_day"]})
    cross = work[cluster].astype(str) + "|" + work["_day"].astype(str)
    both = ols.fit(cov_type="cluster", cov_kwds={"groups": cross})
    params = firm.params
    cov = firm.cov_params().loc[params.index, params.index]
    cov = cov + by_day.cov_params().loc[params.index, params.index]
    cov = cov - both.cov_params().loc[params.index, params.index]
    packed = _from_statsmodels(firm, title, fe, f"{cluster} + event_date", g_firm)
    df = max(min(g_firm, g_day) - 1, 1)
    for name in list(packed["b"]):
        if name not in cov.index:
            continue
        variance = float(cov.loc[name, name])
        if not np.isfinite(variance) or variance <= 0:
            packed["se"][name] = np.nan
            packed["p"][name] = np.nan
            continue
        se = float(np.sqrt(variance))
        packed["se"][name] = se
        t_stat = packed["b"][name] / se if se > 0 else np.nan
        packed["p"][name] = float(2 * sps.t.sf(abs(t_stat), df)) if np.isfinite(t_stat) else np.nan
    packed["g"] = int(min(g_firm, g_day))
    packed["engine"] = "cgm-twoway"
    packed["two_way"] = True
    return packed


def fit_car_regressions(
    events: pd.DataFrame,
    *,
    regression_window: tuple[int, int],
    covariates: list[str],
    cluster: str,
    return_scale: float,
    notes: list[str] | None = None,
    two_way: bool = False,
    wild_draws: int = 0,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    suffix = window_suffix(*regression_window)
    outcome = f"car_{suffix}"
    if events.empty or outcome not in events.columns:
        return []
    columns = ["event_id", "firm_id", "industry", "year", "event_date", "event_date_aligned", outcome, *covariates]
    columns = [col for col in columns if col in events.columns]
    work = events[columns].copy()
    n_input = len(work)
    work["car"] = pd.to_numeric(work[outcome], errors="coerce") * return_scale
    for column in covariates:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["car", *covariates]).reset_index(drop=True)
    if notes is not None and len(work) < n_input:
        notes.append(f"回归因 CAR 或解释变量缺失删除了 {n_input - len(work)} 行。")
    if len(work) < 8:
        if notes is not None:
            notes.append("通过缺失值筛选后不足 8 行，没有回归表。")
        return []
    if cluster not in work.columns:
        if notes is not None:
            notes.append(f"聚类列 {cluster} 不在事件表里，改用 firm_id。")
        cluster = "firm_id"
    work[cluster] = work[cluster].astype(str)
    n_clusters = int(work[cluster].nunique())
    if notes is not None and n_clusters < 30:
        notes.append(f"有效聚类数是 {n_clusters}。聚类较少时，聚类标准误可能偏小。")
    if notes is not None and n_clusters == len(work):
        notes.append("每条公司-事件都是单独的聚类，这时公司聚类接近按行计算的稳健标准误。")
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
    singleton_noted = False
    wild_rng = rng or np.random.default_rng(0)
    for index, (rhs, fe) in enumerate(formulas, start=1):
        sample = _singleton_free(work, fe) if fe else work
        if fe and notes is not None and not singleton_noted and len(sample) < len(work):
            notes.append("固定效应里的单例组会在对应规格中剔除，所以各列的观测数可能不同。")
            singleton_noted = True
        if len(sample) < 8:
            continue
        formula = "car ~ " + rhs
        if fe:
            formula += " | " + " + ".join(fe)
        title = f"({index})"
        try:
            fitted = _fit_pyfixest(formula, sample, cluster)
            packed = _extract(fitted, title, fe, cluster, "pyfixest", len(sample))
        except Exception:
            sm_formula = "car ~ " + rhs
            if fe:
                sm_formula += " + " + " + ".join(f"C({col})" for col in fe)
            fitted = _fit_statsmodels(sm_formula, sample, cluster)
            packed = _from_statsmodels(fitted, title, fe, cluster, sample[cluster].nunique())
        packed["two_way"] = False
        if wild_draws and covariates and not fe and rhs != "1":
            try:
                packed["wild_p"] = wild_cluster_p(sample, covariates, cluster, wild_rng, int(wild_draws))
            except Exception as exc:
                if notes is not None:
                    notes.append(f"wild cluster bootstrap 没有算出：{type(exc).__name__}。")
        models.append(packed)
    if two_way and covariates:
        if notes is not None:
            notes.append(
                "双向聚类的标准误等于公司聚类、事件日聚类与交叉分组三者按 V_firm + V_date − V_firm×date 合成。它仍是相关关系的标准误。"
            )
        for rhs, fe in formulas:
            if rhs == "1":
                continue
            sample = _singleton_free(work, fe) if fe else work
            if len(sample) < 8:
                continue
            title = f"({len(models) + 1})"
            try:
                packed = _cgm_pack(sample, rhs, fe, cluster, title)
            except Exception as exc:
                if notes is not None:
                    notes.append(f"有一列双向聚类没有估出来：{type(exc).__name__}。")
                continue
            if packed is not None:
                models.append(packed)
    if notes is not None and any(model.get("wild_p") for model in models):
        notes.append(
            "不含固定效应的规格另给了 wild cluster bootstrap p。聚类数少于 12 时权重用六点分布，否则用 ±1。表里的星号仍来自聚类标准误，这个 p 也不是处理效应的 p 值。"
        )
    return models
