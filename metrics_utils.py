"""
Utility functions for evaluating the performance of hydrological models.

Centralized here so they can be used by both pipeline.py (computation, only once) 
and app.py (display, without ever recomputing), thus avoiding any duplication of 
logic between the two.

"""
from __future__ import annotations

import hydroeval as he
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

DEFAULT_MODEL_COLS = ("y_pred_rf", "y_pred_xgb")
MODEL_LABELS = {"y_pred_rf": "Random Forest", "y_pred_xgb": "XGBoost"}
MODEL_CODES = {"y_pred_rf": "RF", "y_pred_xgb": "XGB"}


def compute_metrics(sim, obs) -> dict:
    """Computes a standard set of hydrological metrics.

    sim: values simulated / predicted by the model
    obs: observed (measured) values
    """
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)

    mask = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[mask], obs[mask]

    keys = ["NSE", "KGE", "RMSE", "PBIAS", "R2", "MAE", "n_obs"]
    if len(obs) == 0 or np.sum(obs) == 0:
        return {k: np.nan for k in keys}

    nse = float(he.evaluator(he.nse, sim, obs)[0])
    kge = float(he.evaluator(he.kge, sim, obs)[0][0])
    rmse = float(np.sqrt(np.mean((sim - obs) ** 2)))
    pbias = float(100 * np.sum(sim - obs) / np.sum(obs))
    r2 = float(r2_score(obs, sim))
    mae = float(np.mean(np.abs(sim - obs)))

    return {
        "NSE": nse,
        "KGE": kge,
        "RMSE": rmse,
        "PBIAS": pbias,
        "R2": r2,
        "MAE": mae,
        "n_obs": int(len(obs)),
    }


def metrics_table(
            predictions: pd.DataFrame,
            obs_col: str = "value",
            model_cols=DEFAULT_MODEL_COLS,
            group_cols=("site_id", "split"),
        ) -> pd.DataFrame:
    """Builds a metrics table: one row per (basin × period × model)."""
    group_cols = tuple(group_cols)
    rows = []
    for keys, group in predictions.groupby(list(group_cols)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for model_col in model_cols:
            if model_col not in group.columns:
                continue
            m = compute_metrics(group[model_col].values, group[obs_col].values)
            row = dict(zip(group_cols, keys))
            row["model"] = MODEL_CODES.get(model_col, model_col)
            row.update(m)
            rows.append(row)
    return pd.DataFrame(rows)


def quality_flag(nse: float) -> str:
    """Common qualitative benchmark in hydrology for interpreting NSE values (at 
    a daily time step; these thresholds are indicative and not a strict standard).
    source: Moriasi, D. N. et. al. (2007). Model evaluation guidelines for systematic 
    quantification of accuracy in watershed simulations. Transactions of the ASABE, 
    50(3), 885-900.
    """
    if pd.isna(nse):
        return "N/A"
    if nse > 0.75:
        return "Very great"
    if nse > 0.65:
        return "Great"
    if nse > 0.50:
        return "Satisfactory"
    return "Insufficient: the model performs only slightly better than the mean"
