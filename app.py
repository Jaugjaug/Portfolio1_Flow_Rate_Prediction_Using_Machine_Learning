"""
Streamlit App — visualization of streamflow predictions by catchment.

This app DOES NOT DOWNLOAD OR TRAIN anything on its own: it reads the files already
computed by pipeline.py (data/processed/*.parquet). Changing the catchment in
the dropdown menu therefore only filters a DataFrame already in memory → nearly
instantaneous.

To (re)generate the data:
    - python pipeline.py in a terminal, or
    - the "🔄 Run / recompute pipeline" button in the sidebar (with a progress bar).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import pipeline
from metrics_utils import MODEL_LABELS, quality_flag

st.set_page_config(page_title="Streamflow Prediction", layout="wide")

PROCESSED_DIR = Path("data/processed")
PRED_PATH = PROCESSED_DIR / "predictions.parquet"
METRICS_PATH = PROCESSED_DIR / "metrics.parquet"

MODEL_COLORS = {"y_pred_rf": "#2E86AB", "y_pred_xgb": "#E07A5F"}
MODEL_CODE_OF_COL = {"y_pred_rf": "RF", "y_pred_xgb": "XGB"}


@st.cache_data(show_spinner=False)
def load_outputs():
    predictions = pd.read_parquet(PRED_PATH)
    predictions["date"] = pd.to_datetime(predictions["date"])
    metrics = pd.read_parquet(METRICS_PATH)
    return predictions, metrics


def run_pipeline_with_progress():
    progress_bar = st.progress(0.0)
    status = st.empty()

    def _cb(fraction, label):
        progress_bar.progress(min(max(fraction, 0.0), 1.0))
        status.write(label)

    with st.spinner("Calculation in progress (download + training)…"):
        pipeline.run_pipeline(progress_callback=_cb)

    status.write("✅ Completed")
    st.cache_data.clear()


def flow_duration_curve(series: pd.Series):
    """Sorts the streamflow values in descending order and calculates their exceedance probability."""
    s = series.dropna().sort_values(ascending=False).reset_index(drop=True)
    exceedance = (s.index + 1) / (len(s) + 1) * 100
    return exceedance, s


# ---------------------------------------------------------------- 
st.sidebar.header("Data")
if st.sidebar.button("🔄 Run / recalculate the pipeline"):
    run_pipeline_with_progress()
    st.rerun()

data_ready = PRED_PATH.exists() and METRICS_PATH.exists()

if not data_ready:
    st.title("Watershed streamflow prediction")
    st.info(
        "No computed data available yet. Run the pipeline for the first time: "
        "button in the sidebar, or python pipeline.py in a terminal.\n\n"
        "⚠️ The initial computation downloads ~20 years of daily streamflow for 20 watersheds "
        "(USGS network calls) and then trains two models: this will take a moment. "
        "Switching watersheds afterwards will be instantaneous."
    )
    st.stop()

predictions, metrics = load_outputs()

# ------------------------------------------------------------------ 
with st.sidebar.expander("📅 Check common periods"):
    st.markdown("Test here whether a common training period is possible for all watersheds.")
    
    test_start_input = st.date_input("Common period start", value=pd.to_datetime("2000-01-08"))
    test_end_input = st.date_input("Common period end", value=pd.to_datetime("2016-12-31"))
    
    coverage_data = []
    temp_basin_list = sorted(predictions["site_id"].unique().tolist())
    
    for b in temp_basin_list:
        b_df = predictions[predictions["site_id"] == b]
        b_min = b_df["date"].min().date()
        b_max = b_df["date"].max().date()
        
        is_covered = (b_min <= test_start_input) and (b_max >= test_end_input)
        coverage_data.append({
            "Watershed": b,
            "Data start date": b_min,
            "Data end date": b_max,
            "Common period valid?": "✅ yes" if is_covered else "❌ Missing data"
        })
    
    df_coverage = pd.DataFrame(coverage_data)
    st.dataframe(df_coverage, use_container_width=True, hide_index=True)
    
basin_list = sorted(predictions["site_id"].unique().tolist())
basin = st.sidebar.selectbox("Watershed", basin_list)

model_choice = st.sidebar.radio("Displayed model", ["Random Forest", "XGBoost", "Both"], index=2)
model_cols = {
    "Random Forest": ["y_pred_rf"],
    "XGBoost": ["y_pred_xgb"],
    "Both": ["y_pred_rf", "y_pred_xgb"],
}[model_choice]

split_choice = st.sidebar.radio(
    "Displayed period",
    ["Validation (test set, unseen by the model)", "Calibration (train)", "Full series"],
    index=0,
)

# --------------------------------------------------------------------- 
basin_data = predictions[predictions["site_id"] == basin].sort_values("date")
if split_choice.startswith("Validation"):
    basin_view = basin_data[basin_data["split"] == "test"]
elif split_choice.startswith("Calibration"):
    basin_view = basin_data[basin_data["split"] == "train"]
else:
    basin_view = basin_data

# ---------------------------------------------------------------------- 
st.title("Watershed streamflow prediction")
st.caption(f"Selected watershed : **{basin}** — {len(basin_view)} displayed days")

# ------------------------------------------------------------------- 
st.subheader("Model performance (calculated over the validation period)")
basin_metrics_test = metrics[(metrics["site_id"] == basin) & (metrics["split"] == "test")]

if basin_metrics_test.empty:
    st.warning("No metrics available for this watershed — re-run the pipeline.")
else:
    for col in model_cols:
        code = MODEL_CODE_OF_COL[col]
        row = basin_metrics_test[basin_metrics_test["model"] == code]
        if row.empty:
            continue
        row = row.iloc[0]
        st.markdown(f"**{MODEL_LABELS[col]}**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("NSE", f"{row['NSE']:.2f}", help=quality_flag(row["NSE"]))
        m2.metric("KGE", f"{row['KGE']:.2f}")
        m3.metric("PBIAS", f"{row['PBIAS']:.1f} %", help="Bias : >0 = model overestimates, <0 = underestimes")
        m4.metric("RMSE", f"{row['RMSE']:.1f} cfs")

    with st.expander("View details (RMSE, MAE, R², and train vs. test comparison)"):
        st.dataframe(
            metrics[metrics["site_id"] == basin].sort_values(["split", "model"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "If the training score is significantly better than the test score, "
            "the model is overfitting; if they are close, it generalizes well."
        )

# --------------------------------------------------------------------- 
tab_series, tab_parity, tab_resid, tab_fdc = st.tabs(
    ["📈 Time series","🎯 Observed vs. predicted","📉 Residuals","📊 Flow duration curve"])

with tab_series:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=basin_view["date"], y=basin_view["value"], name="Observed",
        line=dict(color="black", width=1.5),
    ))
    for col in model_cols:
        fig.add_trace(go.Scatter(
            x=basin_view["date"], y=basin_view[col], name=MODEL_LABELS[col],
            line=dict(color=MODEL_COLORS[col], width=1),
        ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Streamflow (cfs)",
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=460, margin=dict(t=30),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Use the slider below the chart to zoom in on a specific period.")

with tab_parity:
    fig = go.Figure()
    for col in model_cols:
        fig.add_trace(go.Scatter(
            x=basin_view["value"], y=basin_view[col], mode="markers", name=MODEL_LABELS[col],
            marker=dict(color=MODEL_COLORS[col], size=4, opacity=0.5),
        ))
    max_val = float(basin_view["value"].max()) if len(basin_view) else 1.0
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val], mode="lines", name="1:1 (perfect)",
        line=dict(color="gray", dash="dash"),
    ))
    fig.update_layout(
        xaxis_title="Observed streamflow (cfs)", yaxis_title="Predicted streamflow (cfs)",
        height=460, margin=dict(t=30),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The closer the points are to the diagonal, the better the prediction.")

with tab_resid:
    fig = go.Figure()
    for col in model_cols:
        residual = basin_view[col] - basin_view["value"]
        fig.add_trace(go.Scatter(
            x=basin_view["date"], y=residual, name=MODEL_LABELS[col],
            line=dict(color=MODEL_COLORS[col], width=1),
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Residual = predicted − observed (cfs)",
        height=420, margin=dict(t=30),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Residuals that drift seasonally indicate a systematic model bias.")

with tab_fdc:
    fig = go.Figure()
    exc, s = flow_duration_curve(basin_view["value"])
    fig.add_trace(go.Scatter(x=exc, y=s, name="Observed", line=dict(color="black")))
    for col in model_cols:
        exc, s = flow_duration_curve(basin_view[col])
        fig.add_trace(go.Scatter(x=exc, y=s, name=MODEL_LABELS[col], line=dict(color=MODEL_COLORS[col])))
    fig.update_layout(
        xaxis_title="% of time streamflow is exceeded", yaxis_title="Streamflow (cfs, log scale)",
        yaxis_type="log", height=460, margin=dict(t=30),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Standard hydrological diagnostic: checks whether the model accurately" 
               "reproduces high flows (left) and low flows (right), not just the mean.")

# with st.expander("ℹ️ Why only a CAMELS attributes file, not the full CAMELS-US dataset?"):
#     st.markdown(
#         "The pipeline reads only a single small text file (camels_topo.txt, a few "
#         "hundred KB) containing the drainage area of each watershed to select 20 "
#         "medium-sized basins and retrieve their USGS IDs. This is not the entire "
#         "CAMELS-US dataset (which also contains daily meteorological forcing data "
#         "for ~670 basins, spanning several GB). The streamflow data itself is downloaded "
#         "directly from the USGS service, not from CAMELS."
#     )
