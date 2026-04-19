"""
dashboard/app.py
================
Streamlit dashboard for Customer Churn Prediction.

Features:
  - Interactive customer feature input
  - Real-time churn probability prediction (via API or direct model)
  - Gauge chart for churn probability
  - Risk tier indicator
  - Feature importance visualization
  - Batch CSV upload and analysis
  - Business insights panel
"""

import io
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yaml

# Ensure project root in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Page config (MUST be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Predictor Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  /* Import Google Font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Main background */
  .stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0ff;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.08);
  }

  /* Metric cards */
  .metric-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s;
  }
  .metric-card:hover { transform: translateY(-3px); }
  .metric-value { font-size: 2rem; font-weight: 700; }
  .metric-label { font-size: 0.85rem; color: #a0a0c0; margin-top: 4px; }

  /* Risk badge */
  .badge-high   { background: #ff4b4b22; color: #ff4b4b; border: 1px solid #ff4b4b44; border-radius: 8px; padding: 4px 12px; font-weight: 600; }
  .badge-medium { background: #ffa50022; color: #ffa500; border: 1px solid #ffa50044; border-radius: 8px; padding: 4px 12px; font-weight: 600; }
  .badge-low    { background: #00c85322; color: #00c853; border: 1px solid #00c85344; border-radius: 8px; padding: 4px 12px; font-weight: 600; }

  /* Section headers */
  .section-header {
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #c4b5fd;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
  }

  /* Button overrides */
  .stButton > button {
    background: linear-gradient(135deg, #4361ee, #7209b7);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 28px;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
    transition: all 0.2s;
  }
  .stButton > button:hover { opacity: 0.9; transform: translateY(-2px); }

  /* Selectbox / text input */
  .stSelectbox > div > div, .stNumberInput input {
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.15) !important;
    color: #e0e0ff !important;
    border-radius: 8px !important;
  }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@st.cache_resource
def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_model_direct():
    """Load model directly (offline mode, no API needed)."""
    import joblib
    from src.preprocessing import load_config as _lc
    cfg = _lc()
    model_path = cfg["api"]["model_path"]
    if Path(model_path).exists():
        return joblib.load(model_path), cfg
    return None, cfg


CONFIG = load_config()
API_URL = CONFIG["dashboard"]["api_url"]


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_via_api(data: dict) -> dict:
    """Call the FastAPI endpoint."""
    try:
        resp = requests.post(f"{API_URL}/predict", json=data, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def predict_direct(data: dict) -> dict:
    """Predict using loaded model directly (no API server required)."""
    from src.predict import predict_single
    model, cfg = load_model_direct()
    if model is None:
        return {"error": "Model not trained yet. Run main_pipeline.py first."}
    return predict_single(data, model=model, config=cfg)


def get_prediction(data: dict, use_api: bool) -> dict:
    if use_api:
        return predict_via_api(data)
    return predict_direct(data)


# ---------------------------------------------------------------------------
# Gauge chart
# ---------------------------------------------------------------------------

def draw_gauge(probability: float) -> plt.Figure:
    """Draw a semi-circular gauge for churn probability."""
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    # Background arc
    theta_bg = np.linspace(np.pi, 0, 200)
    ax.plot(theta_bg, [1] * 200, color="#2a2a4a", linewidth=18, solid_capstyle="round")

    # Value arc
    theta_val = np.linspace(np.pi, np.pi - probability * np.pi, 200)
    color = "#ff4b4b" if probability >= 0.7 else ("#ffa500" if probability >= 0.4 else "#00c853")
    ax.plot(theta_val, [1] * 200, color=color, linewidth=18, solid_capstyle="round")

    # Center text
    ax.text(0, 0, f"{probability:.1%}", ha="center", va="center",
            fontsize=22, fontweight="bold", color=color, transform=ax.transData)
    ax.text(0, -0.4, "Churn Probability", ha="center", va="center",
            fontsize=9, color="#a0a0c0", transform=ax.transData)

    ax.set_ylim(0, 1.3)
    ax.set_xlim(0, 2 * np.pi)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ---------------------------------------------------------------------------
# Sidebar — Customer Input Form
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    st.sidebar.markdown("## 👤 Customer Profile")
    st.sidebar.markdown("---")

    with st.sidebar.expander("📋 Demographics", expanded=True):
        gender = st.selectbox("Gender", ["Female", "Male"], key="gender")
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No", key="senior")
        partner = st.selectbox("Partner", ["Yes", "No"], key="partner")
        dependents = st.selectbox("Dependents", ["Yes", "No"], key="dependents")

    with st.sidebar.expander("📅 Account Info", expanded=True):
        tenure = st.slider("Tenure (months)", 0, 72, 12, key="tenure")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract")
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"], key="paperless")
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], key="payment")
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 85.0, step=0.5, key="monthly")
        total = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly * max(tenure, 1), step=1.0, key="total")

    with st.sidebar.expander("📱 Services", expanded=False):
        phone = st.selectbox("Phone Service", ["Yes", "No"], key="phone")
        multi = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key="multi")
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], key="internet")
        security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], key="security")
        backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], key="backup")
        device = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], key="device")
        tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], key="tech")
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], key="tv")
        movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], key="movies")

    return {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multi, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": device, "TechSupport": tech,
        "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": total,
    }


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def main():
    # Header
    st.markdown("""
    <div style="text-align:center; padding: 30px 0 10px 0;">
      <h1 style="font-size:2.5rem; font-weight:800; background:linear-gradient(90deg,#4361ee,#f72585,#7209b7);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        📊 Churn Prediction Dashboard
      </h1>
      <p style="color:#a0a0c0; font-size:1rem;">
        AI-powered customer churn prediction for telecom businesses
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Mode selector
    col_mode1, col_mode2 = st.columns([3, 1])
    with col_mode2:
        use_api = st.toggle("🔌 Use API Mode", value=False,
                            help="Toggle ON to use FastAPI (requires running server). OFF uses model directly.")

    # Get customer data from sidebar
    customer_data = render_sidebar()

    # Predict button
    st.sidebar.markdown("---")
    predict_btn = st.sidebar.button("🔍 Predict Churn", key="predict_btn")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📂 Batch Analysis", "📈 Model Insights"])

    # ── TAB 1: Single Prediction ──────────────────────────────────────────
    with tab1:
        if predict_btn:
            with st.spinner("Analyzing customer data …"):
                result = get_prediction(customer_data, use_api)

            if "error" in result:
                st.error(f"❌ Prediction failed: {result['error']}")
            else:
                prob = result["churn_probability"]
                label = result["churn_label_text"]
                risk = result["risk_tier"]
                conf = result["confidence"]

                # ── Result header ──
                st.markdown("---")
                col_gauge, col_info = st.columns([1, 2])

                with col_gauge:
                    fig_gauge = draw_gauge(prob)
                    st.pyplot(fig_gauge, use_container_width=True)
                    plt.close()

                with col_info:
                    st.markdown(f"""
                    <div style="padding: 20px;">
                      <div class="section-header">Prediction Result</div>
                      <div style="font-size:2rem; font-weight:800; color:{'#ff4b4b' if label=='Churn' else '#00c853'}; margin:8px 0;">
                        {'🔴' if label=='Churn' else '🟢'} {label}
                      </div>
                      <div style="margin:12px 0;">
                        Risk Tier: <span class="badge-{'high' if risk=='High' else 'medium' if risk=='Medium' else 'low'}">{risk}</span>
                      </div>
                      <div style="color:#a0a0c0; font-size:0.9rem; margin-top:12px;">
                        Model confidence: <strong style="color:#e0e0ff;">{conf:.1%}</strong>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── KPI metrics ──
                st.markdown("---")
                c1, c2, c3, c4 = st.columns(4)
                kpis = [
                    (f"{prob:.1%}", "Churn Probability", "#4361ee"),
                    (risk, "Risk Tier", "#f72585" if risk == "High" else "#ffa500" if risk == "Medium" else "#00c853"),
                    (f"${customer_data['MonthlyCharges']:.0f}/mo", "Monthly Charges", "#7209b7"),
                    (f"{customer_data['tenure']} mo", "Tenure", "#06d6a0"),
                ]
                for col, (val, lbl, color) in zip([c1, c2, c3, c4], kpis):
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                          <div class="metric-value" style="color:{color};">{val}</div>
                          <div class="metric-label">{lbl}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # ── Key customer signals ──
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">📌 Key Risk Signals</div>', unsafe_allow_html=True)
                signals = []
                if customer_data["Contract"] == "Month-to-month":
                    signals.append("⚠️ Month-to-month contract (high churn correlation)")
                if customer_data["tenure"] < 12:
                    signals.append("⚠️ Low tenure — early churn risk window")
                if customer_data["MonthlyCharges"] > 80:
                    signals.append("💸 High monthly charges may drive dissatisfaction")
                if customer_data["OnlineSecurity"] == "No" and customer_data["InternetService"] != "No":
                    signals.append("🔒 No online security service")
                if customer_data["TechSupport"] == "No" and customer_data["InternetService"] != "No":
                    signals.append("🛠️ No tech support subscription")
                if customer_data["PaymentMethod"] == "Electronic check":
                    signals.append("💳 Electronic check payment (historically higher churn)")

                if signals:
                    for s in signals:
                        st.markdown(f"- {s}")
                else:
                    st.markdown("✅ No major risk signals detected for this customer.")

        else:
            st.markdown("""
            <div style="text-align:center; padding:60px 20px; color:#a0a0c0;">
              <div style="font-size:3rem;">👈</div>
              <div style="font-size:1.1rem; margin-top:12px;">
                Fill in the customer details in the sidebar and click <strong>Predict Churn</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 2: Batch Analysis ─────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">📂 Batch CSV Upload</div>', unsafe_allow_html=True)
        st.markdown("Upload a CSV file with the same columns as the Telco dataset to get batch predictions.")

        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")

        if uploaded:
            df_upload = pd.read_csv(uploaded)
            st.markdown(f"**Loaded {len(df_upload)} records** — {df_upload.shape[1]} columns")
            st.dataframe(df_upload.head(5), use_container_width=True)

            if st.button("🚀 Run Batch Prediction", key="batch_predict"):
                with st.spinner(f"Predicting for {len(df_upload)} customers …"):
                    from src.predict import predict_batch as _pb
                    model, cfg = load_model_direct()
                    if model is None:
                        st.error("Model not found. Train it first.")
                    else:
                        results = _pb(df_upload, model=model, config=cfg)

                        n_churn = int(results["churn_label"].sum())
                        churn_rate = n_churn / len(results) * 100

                        # Summary KPIs
                        b1, b2, b3 = st.columns(3)
                        with b1:
                            st.metric("Total Customers", len(results))
                        with b2:
                            st.metric("Predicted to Churn", n_churn,
                                      delta=f"{churn_rate:.1f}% of batch",
                                      delta_color="inverse")
                        with b3:
                            if "MonthlyCharges" in results:
                                risk_rev = results[results["churn_label"] == 1]["MonthlyCharges"].sum()
                                st.metric("Revenue at Risk", f"${risk_rev:,.0f}/mo")

                        # Segmented table
                        st.markdown("---")
                        st.markdown("**High-Risk Customers (Prob ≥ 0.70)**")
                        high_risk = results[results["churn_probability"] >= 0.70].sort_values(
                            "churn_probability", ascending=False
                        )
                        st.dataframe(
                            high_risk[["churn_probability", "churn_label", "risk_tier"]
                                      + (["MonthlyCharges", "tenure", "Contract"] if all(
                                          c in high_risk.columns for c in ["MonthlyCharges", "tenure", "Contract"]
                                      ) else [])].head(20),
                            use_container_width=True
                        )

                        # Download
                        csv_bytes = results.to_csv(index=False).encode()
                        st.download_button(
                            "⬇️ Download Full Results",
                            data=csv_bytes,
                            file_name="churn_predictions.csv",
                            mime="text/csv",
                        )

    # ── TAB 3: Model Insights ─────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">📈 Model Performance & Insights</div>', unsafe_allow_html=True)

        # Load saved metrics
        metrics_path = Path("reports/evaluation_metrics.json")
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

            # Model comparison bar chart
            model_names = [k for k in metrics.keys() if k != "business_insights"]
            if model_names:
                metric_keys = ["roc_auc", "f1", "precision", "recall"]
                col_chart, col_table = st.columns([3, 2])

                with col_chart:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    fig.patch.set_facecolor("#0f0c29")
                    ax.set_facecolor("#0f0c29")
                    x = np.arange(len(model_names))
                    width = 0.2
                    colors = ["#4361ee", "#f72585", "#7209b7", "#06d6a0"]
                    for i, (mk, color) in enumerate(zip(metric_keys, colors)):
                        vals = [metrics[m].get(mk, 0) for m in model_names]
                        ax.bar(x + i * width - 0.3, vals, width, label=mk.upper(), color=color, alpha=0.85)
                    ax.set_xticks(x)
                    ax.set_xticklabels([m.replace("_", "\n").title() for m in model_names], color="white")
                    ax.set_ylim([0, 1.15])
                    ax.set_ylabel("Score", color="white")
                    ax.tick_params(colors="white")
                    ax.spines[:].set_color("#444")
                    ax.legend(fontsize=8, facecolor="#1a1a3a", labelcolor="white")
                    ax.set_title("Model Comparison", color="white", fontweight="bold")
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                with col_table:
                    rows = []
                    for m in model_names:
                        rows.append({
                            "Model": m.replace("_", " ").title(),
                            "ROC-AUC": f"{metrics[m].get('roc_auc', 0):.4f}",
                            "F1": f"{metrics[m].get('f1', 0):.4f}",
                            "Precision": f"{metrics[m].get('precision', 0):.4f}",
                            "Recall": f"{metrics[m].get('recall', 0):.4f}",
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Business insights
            if "business_insights" in metrics:
                bi = metrics["business_insights"]
                st.markdown("---")
                st.markdown('<div class="section-header">💼 Business Insights</div>', unsafe_allow_html=True)

                i1, i2 = st.columns(2)
                with i1:
                    st.metric("High-Risk Customers", bi.get("high_risk_count", "—"),
                              delta=f"{bi.get('high_risk_pct', 0):.1f}% of test set",
                              delta_color="inverse")
                with i2:
                    if "monthly_revenue_at_risk" in bi:
                        st.metric("Monthly Revenue at Risk",
                                  f"${bi['monthly_revenue_at_risk']:,.0f}")

                # Top drivers
                if "top_churn_drivers" in bi:
                    st.markdown("**Top Churn Drivers**")
                    drivers_df = pd.DataFrame(bi["top_churn_drivers"])
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    fig2.patch.set_facecolor("#0f0c29")
                    ax2.set_facecolor("#0f0c29")
                    ax2.barh(drivers_df["feature"][::-1], drivers_df["importance"][::-1],
                             color="#7209b7", edgecolor="none")
                    ax2.tick_params(colors="white")
                    ax2.spines[:].set_color("#444")
                    ax2.set_xlabel("Importance", color="white")
                    ax2.set_title("Top 10 Churn Drivers", color="white", fontweight="bold")
                    st.pyplot(fig2, use_container_width=True)
                    plt.close()

            # Saved plots
            plots_dir = Path("reports/figures")
            if plots_dir.exists():
                st.markdown("---")
                st.markdown('<div class="section-header">📊 Evaluation Plots</div>', unsafe_allow_html=True)
                plot_files = list(plots_dir.glob("*.png"))
                if plot_files:
                    cols = st.columns(2)
                    for i, pf in enumerate(plot_files[:6]):
                        with cols[i % 2]:
                            st.image(str(pf), caption=pf.stem.replace("_", " ").title(),
                                     use_container_width=True)
        else:
            st.info("ℹ️ No evaluation results found. Run `python main_pipeline.py` to train and evaluate models.")


if __name__ == "__main__":
    main()
