import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pipeline import run_pipeline
from src.config import OUTPUT_FILE, RAW_DATA_DIR
from src.feature_engineering import extract_features_from_transactions
from src.preprocessing import prepare_features
from src.model import train_model, score_wallets
from src.output import build_results

st.set_page_config(
    page_title="Smart Contract Behaviour Anomaly Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ChainGuard: Smart Contract Behaviour Anomaly Detector")
st.markdown(
    "Unsupervised ML detection of abnormal on-chain transaction patterns using **Isolation Forest** on Ethereum Mainnet data."
)

# Sidebar navigation and contract presets
st.sidebar.header("Configuration & Analysis Mode")
mode = st.sidebar.radio(
    "Choose Data Source:",
    ["📊 Pre-computed Results", "🌐 Live Etherscan Contract Analysis", "📁 Upload Custom Transactions CSV"],
)

PRESET_CONTRACTS = {
    "Uniswap V2: Router 02": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
    "USDC (USD Coin)": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "Tether USD (USDT)": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    "Wrapped Ether (WETH)": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
}

results_df = None

# Mode 1: Pre-computed Results
if mode == "📊 Pre-computed Results":
    st.sidebar.info("Loading pre-trained model analysis from `outputs/wallet_anomaly_results.csv`.")
    if os.path.exists(OUTPUT_FILE):
        results_df = pd.read_csv(OUTPUT_FILE)
    else:
        st.warning("`outputs/wallet_anomaly_results.csv` not found. Running pipeline to generate it...")
        results_df = run_pipeline()

# Mode 2: Live Contract Analysis
elif mode == "🌐 Live Etherscan Contract Analysis":
    st.subheader("🌐 Live On-Chain Contract Analyzer")
    selected_preset = st.selectbox("Select Preset Contract:", ["Custom Address"] + list(PRESET_CONTRACTS.keys()))

    if selected_preset == "Custom Address":
        contract_input = st.text_input("Enter Smart Contract Address (0x...):", value="").strip()
    else:
        contract_input = PRESET_CONTRACTS[selected_preset]
        st.text_input("Smart Contract Address:", value=contract_input, disabled=True)

    days_to_fetch = st.slider("Historical Time Window (Days):", min_value=7, max_value=180, value=90, step=7)

    if st.button("🚀 Analyze Smart Contract Behaviour", type="primary"):
        if not contract_input or not contract_input.startswith("0x") or len(contract_input) != 42:
            st.error("Please enter a valid 42-character Ethereum contract address starting with 0x.")
        else:
            with st.spinner(f"Fetching transactions from Etherscan V2 and running Isolation Forest for {contract_input}..."):
                try:
                    results_df = run_pipeline(contract_address=contract_input, days=days_to_fetch)
                    st.success(f"Analysis complete for {contract_input}!")
                except Exception as e:
                    st.error(f"Pipeline execution failed: {e}")

# Mode 3: Upload Custom Transactions CSV
elif mode == "📁 Upload Custom Transactions CSV":
    st.subheader("📁 Upload Raw Transactions CSV")
    st.markdown("Upload any CSV containing raw transactions (with columns: `from`, `value`, `gas`, `gasUsed`, `functionName`, `isError`).")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_input_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded {len(raw_input_df)} raw transaction rows.")
            if st.button("Run Anomaly Detector on Uploaded Data", type="primary"):
                with st.spinner("Extracting features & scoring anomalies..."):
                    features_df = extract_features_from_transactions(raw_input_df)
                    X = prepare_features(features_df)
                    model = train_model(X)
                    scores, flags = score_wallets(model, X)
                    results_df = build_results(features_df, scores, flags)
                    st.success("Analysis finished!")
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")


# Render Dashboard if results are ready
if results_df is not None and not results_df.empty:
    st.divider()

    # Top Metric KPI Cards
    total_wallets = len(results_df)
    anomalies_flagged = int(results_df["is_anomaly"].sum())
    high_risk = int((results_df.get("risk_level") == "HIGH").sum())
    med_risk = int((results_df.get("risk_level") == "MEDIUM").sum())
    low_risk = int((results_df.get("risk_level") == "LOW").sum())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Wallets Profiled", f"{total_wallets:,}")
    col2.metric("Flagged Anomalous", f"{anomalies_flagged:,}", delta=f"{anomalies_flagged/total_wallets*100:.1f}% rate", delta_color="inverse")
    col3.metric("🔴 HIGH Risk", f"{high_risk:,}")
    col4.metric("🟡 MEDIUM Risk", f"{med_risk:,}")
    col5.metric("🟢 LOW Risk", f"{low_risk:,}")

    # Visualizations
    st.subheader("📈 Behavioural Analytics & Distributions")
    tab1, tab2 = st.tabs(["Risk & Anomaly Distributions", "Feature Correlations"])

    with tab1:
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            st.markdown("**Risk Level Breakdown**")
            fig, ax = plt.subplots(figsize=(6, 4))
            risk_counts = results_df["risk_level"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"]).fillna(0)
            colors = ["#dc2626", "#eab308", "#22c55e"]
            ax.bar(risk_counts.index, risk_counts.values, color=colors, edgecolor="black", alpha=0.85)
            ax.set_ylabel("Number of Wallets")
            ax.set_title("Wallets by Risk Category")
            for i, v in enumerate(risk_counts.values):
                ax.text(i, v + (max(risk_counts.values)*0.01), f"{int(v)}", ha="center", fontweight="bold")
            st.pyplot(fig)

        with vcol2:
            st.markdown("**Isolation Forest Anomaly Score Distribution**")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(results_df["anomaly_score"], bins=30, color="#2563eb", edgecolor="black", alpha=0.7)
            ax.axvline(0.60, color="#dc2626", linestyle="--", linewidth=1.5, label="High Risk Threshold (0.60)")
            ax.axvline(0.50, color="#eab308", linestyle="--", linewidth=1.5, label="Medium Risk Threshold (0.50)")
            ax.set_xlabel("Anomaly Score (Higher = More Anomalous)")
            ax.set_ylabel("Wallet Frequency")
            ax.legend(loc="upper right", fontsize="small")
            st.pyplot(fig)

    with tab2:
        scol1, scol2 = st.columns(2)
        with scol1:
            st.markdown("**Anomaly Score vs Total Value (ETH)**")
            fig, ax = plt.subplots(figsize=(6, 4))
            scatter = ax.scatter(
                np.log1p(results_df["total_value"].clip(lower=0)),
                results_df["anomaly_score"],
                c=results_df["is_anomaly"].map({True: "#dc2626", False: "#3b82f6"}),
                alpha=0.6,
                edgecolors="none",
            )
            ax.set_xlabel("log(1 + Total ETH Transferred)")
            ax.set_ylabel("Anomaly Score")
            st.pyplot(fig)

        with scol2:
            st.markdown("**Gas Efficiency vs Failure Rate**")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(
                results_df["gas_efficiency"],
                results_df["failure_rate"],
                c=results_df["is_anomaly"].map({True: "#dc2626", False: "#10b981"}),
                alpha=0.6,
                edgecolors="none",
            )
            ax.set_xlabel("Gas Efficiency (Gas Used / Allocated)")
            ax.set_ylabel("Transaction Failure Rate")
            st.pyplot(fig)

    # Filterable Data Table
    st.subheader("🔍 Wallet Anomaly Dossier & Inspection")
    fcol1, fcol2, fcol3 = st.columns([1, 1, 2])
    with fcol1:
        risk_filter = st.multiselect("Filter by Risk Level:", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"])
    with fcol2:
        flag_filter = st.selectbox("Anomaly Flag:", ["All", "Only Flagged (is_anomaly=True)", "Cleared Normal"])
    with fcol3:
        search_addr = st.text_input("Search Wallet Address (0x...):", value="").strip().lower()

    filtered_df = results_df.copy()
    if risk_filter:
        filtered_df = filtered_df[filtered_df["risk_level"].isin(risk_filter)]
    if flag_filter == "Only Flagged (is_anomaly=True)":
        filtered_df = filtered_df[filtered_df["is_anomaly"] == True]
    elif flag_filter == "Cleared Normal":
        filtered_df = filtered_df[filtered_df["is_anomaly"] == False]
    if search_addr:
        filtered_df = filtered_df[filtered_df["from"].str.lower().str.contains(search_addr)]

    st.write(f"Showing **{len(filtered_df)}** matching wallets:")
    display_columns = [
        "rank",
        "from",
        "risk_level",
        "anomaly_score",
        "is_anomaly",
        "transaction_count",
        "total_value",
        "average_value",
        "failed_transactions",
        "failure_rate",
        "gas_efficiency",
    ]
    st.dataframe(
        filtered_df[[c for c in display_columns if c in filtered_df.columns]],
        use_container_width=True,
        hide_index=True,
    )

    # CSV Download Button
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Results (CSV)",
        data=csv_bytes,
        file_name="smart_contract_anomaly_results.csv",
        mime="text/csv",
    )

