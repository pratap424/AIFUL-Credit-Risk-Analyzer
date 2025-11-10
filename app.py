# ======================================================
# app_flagship.py — AIFUL Credit Risk Analyzer (Flagship Edition)
# ======================================================

# ---------- Imports ----------
import streamlit as st
st.set_page_config(page_title="AIFUL Credit Risk Analyzer", layout="wide")

import pandas as pd
import numpy as np
import joblib, json, shap, plotly.express as px
import plotly.graph_objects as go
import warnings, io
from fpdf import FPDF
from sklearn.cluster import KMeans
from sklearn.compose import _column_transformer

# ---------- Compatibility Patch ----------
if not hasattr(_column_transformer, "_RemainderColsList"):
    class _RemainderColsList(list): pass
    _column_transformer._RemainderColsList = _RemainderColsList

# ---------- Load Artifacts ----------
@st.cache_resource
def load_assets():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        model = joblib.load("model_pipeline.pkl")
    with open("feature_list.json") as f:
        features = json.load(f)
    preprocessor = joblib.load("preprocessor.pkl")
    test = pd.read_csv("test.csv")
    return model, preprocessor, features, test

clf, preprocessor, features, test = load_assets()

# ---------- Helper Functions ----------
def predict_customer_risk(customer_row):
    proba = clf.predict_proba(customer_row)[0][1]
    if proba < 0.3:
        tier = "🟢 Low Risk"
    elif proba < 0.6:
        tier = "🟠 Medium Risk"
    else:
        tier = "🔴 High Risk"
    return proba, tier

def recommended_limit(original_limit, risk):
    if risk < 0.3: factor = 1.0
    elif risk < 0.6: factor = 0.8
    else: factor = 0.5
    return int(original_limit * factor)

@st.cache_resource
def shap_explainer(_model):
    return shap.TreeExplainer(_model.named_steps["model"])

# ---------- Header ----------
st.markdown(
    """
    <div style="text-align:center; padding:15px;">
        <h1 style="color:#0066cc;">💳 AIFUL AI Credit Risk Platform</h1>
        <h4>Predict • Explain • Simulate • Act — Responsible AI for Smarter Lending</h4>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- Sidebar: What-If Simulator ----------
st.sidebar.header("🧩 What-If Risk Simulator")
inc_change = st.sidebar.slider("Adjust Annual Income (%)", -50, 100, 0)
loan_change = st.sidebar.slider("Adjust Loan Amount (%)", -50, 100, 0)
st.sidebar.markdown("---")
st.sidebar.caption("Simulate changes to test how risk scores respond.")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Portfolio Dashboard", 
    "👤 Customer Insights", 
    "⚖️ Fairness & Ethics", 
    "🧮 Business Intelligence"
])

# =================================================
# TAB 1 — PORTFOLIO DASHBOARD
# =================================================
with tab1:
    st.subheader("Portfolio-Level Risk Overview")
    X_test = test[features]
    probs = clf.predict_proba(X_test)[:, 1]
    test["Predicted_Prob"] = probs
    test["Risk_Tier"] = pd.cut(probs, bins=[0, 0.3, 0.6, 1], labels=["Low", "Medium", "High"])

    # Portfolio KPI
    high_risk_pct = (test["Risk_Tier"] == "High").mean() * 100
    medium_risk_pct = (test["Risk_Tier"] == "Medium").mean() * 100
    low_risk_pct = (test["Risk_Tier"] == "Low").mean() * 100
    pri = (high_risk_pct*1.0 + medium_risk_pct*0.5 + low_risk_pct*0.1) / 100

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("🔴 High-Risk Customers", f"{high_risk_pct:.1f}%")
    kpi2.metric("🏦 Portfolio Risk Index", f"{pri:.2f}")
    kpi3.metric("🟢 Low-Risk Customers", f"{low_risk_pct:.1f}%")

    # Risk Distribution
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(
            test, x="Predicted_Prob", nbins=30, color="Risk_Tier",
            title="Default Probability Distribution",
            color_discrete_map={"Low":"green","Medium":"orange","High":"red"}
        )
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        tier_counts = test["Risk_Tier"].value_counts().reset_index()
        tier_counts.columns = ["Risk_Tier", "count"]
        fig2 = px.pie(
            tier_counts, values="count", names="Risk_Tier",
            color="Risk_Tier", title="Risk Tier Composition",
            color_discrete_map={"Low":"green","Medium":"orange","High":"red"}
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Correlation Heatmap
    st.subheader("Feature Influence on Default Risk")
    corr = test.corr(numeric_only=True)["Predicted_Prob"].sort_values(ascending=False).head(10)
    fig_heat = px.bar(
        corr[::-1], x=corr[::-1].values, y=corr[::-1].index, orientation="h",
        color=corr[::-1].values, color_continuous_scale="RdBu_r",
        title="Top 10 Features Most Correlated with Default Probability"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# =================================================
# TAB 2 — CUSTOMER INSIGHTS
# =================================================
with tab2:
    st.subheader("Individual Customer Risk Analysis")

    customer_id = st.selectbox("Select Customer ID", test["ID"].tolist())
    customer_data = test[test["ID"] == customer_id][features]
    orig_row = test.loc[test["ID"] == customer_id]

    # Simulation
    sim_data = customer_data.copy()
    if "Total Annual Income" in sim_data.columns:
        sim_data["Total Annual Income"] *= (1 + inc_change / 100)
    if "Amount of Unsecured Loans" in sim_data.columns:
        sim_data["Amount of Unsecured Loans"] *= (1 + loan_change / 100)

    risk_score, tier = predict_customer_risk(customer_data)
    sim_risk, sim_tier = predict_customer_risk(sim_data)

    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Default Probability", f"{risk_score*100:.2f}%")
    col2.metric("🏷️ Risk Tier", tier)
    col3.metric("💡 Simulated Risk", f"{sim_risk*100:.2f}% ({sim_tier})")

    if "Application Limit Amount(Desired)" in orig_row.columns:
        rec_limit = recommended_limit(orig_row["Application Limit Amount(Desired)"].values[0], risk_score)
        st.metric("💰 Recommended Credit Limit", f"₹ {rec_limit:,}")

    # SHAP Explainability
    st.divider()
    st.subheader("Explainability — Key Drivers")
    explainer = shap_explainer(clf)
    shap_values = explainer.shap_values(preprocessor.transform(customer_data))
    shap_values_final = shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values
    shap_df = pd.DataFrame({
        "Feature": features, "SHAP Value": shap_values_final[0]
    }).sort_values("SHAP Value", key=abs, ascending=False).head(10)

    fig = px.bar(
        shap_df[::-1], x="SHAP Value", y="Feature", orientation="h",
        color="SHAP Value", color_continuous_scale="RdBu_r",
        title="Top 10 Factors Influencing Default Risk"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Natural-Language Summary
    top_feats = shap_df["Feature"].head(3).tolist()
    summary = (
        f"Customer {customer_id} shows {tier} with a default probability of {risk_score*100:.1f}%. "
        f"The major drivers increasing risk are {top_feats[0]} and {top_feats[1]}, "
        f"while {top_feats[2]} slightly offsets risk."
    )
    st.success(summary)

    # Credit Health Report
    if st.button(" Generate Credit Health Report (PDF)"):
        buffer = io.BytesIO()
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, f"Credit Health Report - Customer {customer_id}", ln=True, align="C")
    
        pdf.set_font("Arial", "", 12)
        pdf.ln(10)
        # Clean the emoji out of tier text before writing
        clean_tier = tier.encode('ascii', 'ignore').decode()
        pdf.cell(0, 10, f"Default Probability: {risk_score*100:.2f}%", ln=True)
        pdf.cell(0, 10, f"Risk Tier: {clean_tier}", ln=True)
        pdf.cell(0, 10, f"Recommended Limit: ₹ {rec_limit:,}", ln=True)

        pdf.ln(10)
        pdf.cell(0, 10, "Top Risk Factors:", ln=True)
        for i, row in shap_df.iterrows():
            pdf.cell(0, 8, f"- {row['Feature']}: {row['SHAP Value']:.3f}", ln=True)
        pdf.output(buffer)
        st.download_button(
            "Download Report",
            data=buffer.getvalue(),
            file_name=f"Credit_Report_{customer_id}.pdf",
            mime="application/pdf"
        )

# =================================================
# TAB 3 — FAIRNESS & ETHICS
# =================================================
with tab3:
    st.subheader("Fair Lending & Ethical AI Analysis")

    if "Gender" in test.columns and "Single/Married Status" in test.columns:
        fairness = test.pivot_table(
            values="Predicted_Prob",
            index="Gender",
            columns="Single/Married Status",
            aggfunc="mean"
        )
        fig_fair = px.imshow(
            fairness, text_auto=".2f", color_continuous_scale="RdBu_r",
            title="Fairness Heatmap (Avg Predicted Risk)"
        )
        st.plotly_chart(fig_fair, use_container_width=True)

        # Compute fairness score
        gender_diff = abs(test.groupby("Gender")["Predicted_Prob"].mean().diff().iloc[-1])
        marital_diff = abs(test.groupby("Single/Married Status")["Predicted_Prob"].mean().diff().iloc[-1])
        fairness_score = max(0, 100 - (gender_diff + marital_diff) * 500)
        st.metric("🤝 Fair Lending Score", f"{fairness_score:.1f}/100")

        if fairness_score > 85:
            st.success("✅ Model predictions appear balanced across groups.")
        else:
            st.warning("⚠️ Some demographic bias detected — consider rebalancing training data.")

# =================================================
# TAB 4 — BUSINESS INTELLIGENCE
# =================================================
with tab4:
    st.subheader("Policy & Segment Intelligence")

    # Loan-to-Income insight
    if "Amount of Unsecured Loans" in test.columns and "Total Annual Income" in test.columns:
        test["Loan_to_Income"] = test["Amount of Unsecured Loans"] / (test["Total Annual Income"] + 1)
        threshold = test[test["Risk_Tier"] == "High"]["Loan_to_Income"].mean()
        st.warning(f"💡 Customers with Loan-to-Income ratio > {threshold:.2f} are 2x more likely to default.")

    # Customer segmentation
    st.subheader("Behavioral Segmentation (K-Means)")
    num_cols = test.select_dtypes(include=[np.number]).columns
    kmeans = KMeans(n_clusters=4, random_state=42)
    test["Cluster"] = kmeans.fit_predict(test[num_cols].fillna(0))

    fig_cluster = px.scatter(
        test, x="Total Annual Income", y="Amount of Unsecured Loans",
        color="Cluster", hover_data=["Risk_Tier"],
        title="Customer Segments by Behavior"
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

st.markdown("---")
st.caption("Built by Team AIFUL × IIT Mandi — AIHack India 2025 | Explainable • Ethical • Actionable")
