"""
This script:
    - selects catchments (basins) based on CAMELS attributes,
    - downloads their daily streamflow data from the USGS service,
    - constructs the explanatory variables (seasonality + lagged streamflow),
    - trains a Random Forest and an XGBoost model,
    - saves predictions and metrics to data/processed/.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from dataretrieval import waterdata
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import time

from metrics_utils import metrics_table

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

FEATURES = ["doy_sin", "doy_cos", "flow_lag1", "flow_lag7"]
TEST_START = "2017-01-01"  


def _scaled(progress_callback, start: float, end: float):
    """
    Adapts a global callback f(fraction, label) into a local callback f(current, 
    total, label), remapping the progress to the [start, end] range.
    """
    if progress_callback is None:
        return None

    def _cb(current, total, label):
        frac = start + (current / total) * (end - start)
        progress_callback(frac, label)

    return _cb


def select_basins(n_basins: int = 20, seed: int = 42) -> pd.DataFrame:
    """
    Step 1: selects n_basins medium-sized catchments (100–5,000 km²).
    """
    attrs = pd.read_csv(DATA_DIR / "camels/camels_attributes_v2.0/camels_topo.txt", sep=";")
    selection = attrs[(attrs["area_gages2"] > 100) & (attrs["area_gages2"] < 5000)]
    selection = selection.sample(n_basins, random_state=seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    selection.to_csv(DATA_DIR / "selected_basins.csv", index=False)
    return selection


def download_streamflow(
    site_ids, start="2000-01-01", end="2020-12-31", progress_callback=None
) -> pd.DataFrame:
    """
    Step 2: downloads daily streamflow data (USGS, parameter 00060) for each catchment.
    """
    all_flows = []
    n = len(site_ids)
    for i, site_id in enumerate(site_ids, start=1):
        if progress_callback:
            progress_callback(i - 1, n, f"Streamflow download {site_id} ({i}/{n})")
        df, _meta = waterdata.get_daily(
            monitoring_location_id=f"USGS-{str(site_id).zfill(8)}",
            parameter_code="00060", #for flow
            time=[start, end],
        )
        df = df[df["statistic_id"] == "00003"].copy() #For daily mean flow
        df["site_id"] = site_id
        all_flows.append(df)
        time.sleep(1.5)

    if progress_callback:
        progress_callback(n, n, "Download completed")

    flows = pd.concat(all_flows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    flows.to_parquet(PROCESSED_DIR / "streamflow.parquet")
    return flows


def engineer_features(flows: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3: seasonality features (day of year) + autoregressive features (yesterday's 
    streamflow, streamflow from 7 days ago).
    """
    if "time" not in flows.columns:
        raise KeyError(
            "Column 'time' is missing from the result of waterdata.get_daily "
            f"(received columns : {list(flows.columns)})."
        )
    flows = flows.sort_values(["site_id", "time"]).reset_index(drop=True)
    flows["date"] = pd.to_datetime(flows["time"])
    flows["doy_sin"] = np.sin(2 * np.pi * flows["date"].dt.dayofyear / 365)
    flows["doy_cos"] = np.cos(2 * np.pi * flows["date"].dt.dayofyear / 365)
    flows["flow_lag1"] = flows.groupby("site_id")["value"].shift(1)
    flows["flow_lag7"] = flows.groupby("site_id")["value"].shift(7)
    flows = flows.dropna(subset=FEATURES + ["value"])
    return flows


def train_and_evaluate(flows: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
    """
    Step 4: trains Random Forest and XGBoost once, and returns observed + predicted 
    values for all catchments, with a split column (train / test).
    """
    site_dummies = pd.get_dummies(flows["site_id"], prefix="site")
    X = pd.concat([flows[FEATURES], site_dummies], axis=1)
    y = flows["value"]
    is_test = flows["date"] >= pd.Timestamp(TEST_START)

    X_train, y_train = X[~is_test], y[~is_test]
    X_test = X[is_test]

    if X_train.empty or X_test.empty:
        raise ValueError(
            f"The temporal split at TEST_START={TEST_START} leaves no data on both "
            "sides — check the downloaded date range."
        )

    if progress_callback:
        progress_callback(0, 3, "Random Forest training…")
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    if progress_callback:
        progress_callback(1, 3, "XGBoost training…")
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)

    if progress_callback:
        progress_callback(2, 3, "Prediction over the entire period…")

    predictions = flows[["site_id", "date", "value"]].copy()
    predictions["split"] = np.where(is_test, "test", "train")
    predictions["y_pred_rf"] = rf.predict(X)
    predictions["y_pred_xgb"] = xgb.predict(X)

    if progress_callback:
        progress_callback(3, 3, "Modeling completed")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(PROCESSED_DIR / "predictions.parquet")
    return predictions


def run_pipeline(progress_callback=None) -> pd.DataFrame:
    """Runs all steps sequentially. progress_callback(fraction in [0, 1], label), optional."""
    selection = select_basins()
    flows = download_streamflow(
        selection["gauge_id"], progress_callback=_scaled(progress_callback, 0.0, 0.7)
    )
    flows = engineer_features(flows)
    predictions = train_and_evaluate(
        flows, progress_callback=_scaled(progress_callback, 0.7, 1.0)
    )
    metrics = metrics_table(predictions)
    metrics.to_parquet(PROCESSED_DIR / "metrics.parquet")
    if progress_callback:
        progress_callback(1.0, "Terminé ✅")
    return metrics


if __name__ == "__main__":
    try:
        from tqdm import tqdm

        _pbar = tqdm(total=100, desc="Pipeline", unit="%")

        def _cli_progress(fraction, label):
            _pbar.n = int(fraction * 100)
            _pbar.set_description(label)
            _pbar.refresh()

    except ImportError:  
        def _cli_progress(fraction, label):
            print(f"[{fraction:>4.0%}] {label}", file=sys.stderr)

    result_metrics = run_pipeline(progress_callback=_cli_progress)
    print()
    print(result_metrics.sort_values(["site_id", "split", "model"]).to_string(index=False))