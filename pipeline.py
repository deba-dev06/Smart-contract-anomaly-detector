"""
End-to-End Integration Pipeline for Smart Contract Behaviour Anomaly Detector.

Pipeline flow:
Etherscan API (or raw CSVs) -> Raw Transactions -> Feature Extraction -> Isolation Forest -> Anomaly Score & Risk Level -> Output CSV
"""

import os
import sys
from pathlib import Path
import pandas as pd

from fetch_data import fetch_contract_transactions
from src.config import RAW_DATA_DIR, OUTPUT_DIR, OUTPUT_FILE
from src.feature_engineering import extract_features_from_transactions, process_raw_data_folder
from src.preprocessing import prepare_features
from src.model import train_model, score_wallets
from src.output import build_results, save_results


def run_pipeline(contract_address: str = None, days: int = 90, raw_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Executes the complete end-to-end anomaly detection pipeline.
    
    Args:
        contract_address: Optional Ethereum smart contract address to fetch live data for.
        days: Historical days of transaction history to fetch.
        raw_df: Optional pre-loaded raw transaction DataFrame.
        
    Returns:
        pd.DataFrame containing ranked wallets with anomaly scores and risk levels.
    """
    print("\n=======================================================")
    print(" Smart Contract Behaviour Anomaly Detector Pipeline")
    print("=======================================================\n")

    # Step 1: Obtain raw transactions
    if raw_df is not None and not raw_df.empty:
        print(f"[Step 1/4] Using provided DataFrame with {len(raw_df)} transactions.")
        tx_data = raw_df
    elif contract_address:
        print(f"[Step 1/4] Fetching live transactions from Etherscan for: {contract_address}...")
        tx_data = fetch_contract_transactions(contract_address, days=days)
        if tx_data.empty:
            raise RuntimeError(f"Could not retrieve transactions for contract: {contract_address}")
        print(f"  -> Retrieved {len(tx_data)} transactions.")
    else:
        print(f"[Step 1/4] Loading existing raw transaction datasets from: {RAW_DATA_DIR}...")
        tx_data = None
        # We can directly use process_raw_data_folder for multi-contract evaluation

    # Step 2: Feature Engineering
    print("[Step 2/4] Extracting behavioural features per interacting wallet...")
    if tx_data is not None:
        features_df = extract_features_from_transactions(tx_data)
    else:
        features_df = process_raw_data_folder(str(RAW_DATA_DIR))
    
    print(f"  -> Extracted behavioural profiles for {len(features_df)} unique wallets.")

    # Step 3: Machine Learning Model (Isolation Forest)
    print("[Step 3/4] Running Isolation Forest anomaly detection...")
    X = prepare_features(features_df)
    model = train_model(X)
    scores, flags = score_wallets(model, X)

    # Step 4: Scoring, Ranking & LOW / MEDIUM / HIGH Risk Categorization
    print("[Step 4/4] Generating anomaly scores and assigning risk categories...")
    results_df = build_results(features_df, scores, flags)
    
    output_path = save_results(results_df)
    print(f"  -> Results successfully saved to: {output_path}")

    # Summary report
    high_count = (results_df["risk_level"] == "HIGH").sum()
    med_count = (results_df["risk_level"] == "MEDIUM").sum()
    low_count = (results_df["risk_level"] == "LOW").sum()
    anom_count = int(results_df["is_anomaly"].sum())

    print("\n----------------- PIPELINE SUMMARY -----------------")
    print(f" Total Wallets Profiled : {len(results_df)}")
    print(f" Anomaly Flagged (True) : {anom_count} ({anom_count / len(results_df) * 100:.1f}%)")
    print(f" HIGH Risk Wallets      : {high_count}")
    print(f" MEDIUM Risk Wallets    : {med_count}")
    print(f" LOW Risk Wallets       : {low_count}")
    print("----------------------------------------------------\n")

    print("Top 5 Detected Anomalies:")
    display_cols = ["rank", "from", "anomaly_score", "risk_level", "transaction_count", "total_value"]
    print(results_df[display_cols].head(5).to_string(index=False))
    print()

    return results_df


if __name__ == "__main__":
    # If a contract address is passed via command line argument, use it; otherwise run on raw_data
    target_address = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(contract_address=target_address)

