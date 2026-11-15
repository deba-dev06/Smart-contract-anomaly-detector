import os
import sys
from pipeline import run_pipeline
from src.config import DATA_FILE, OUTPUT_FILE
from src.data_loader import load_data
from src.preprocessing import prepare_features
from src.model import train_model, score_wallets
from src.output import build_results, save_results


def main():
    """
    Main entry point for Smart Contract Behaviour Anomaly Detector.
    If a contract address is provided via CLI argument, it triggers the end-to-end pipeline.
    Otherwise, it checks for existing feature sets or processes raw_data.
    """
    if len(sys.argv) > 1:
        contract_addr = sys.argv[1]
        run_pipeline(contract_address=contract_addr)
        return

    # Check if pre-engineered data file exists
    if DATA_FILE.exists():
        print(f"Loading pre-engineered features from: {DATA_FILE}")
        df = load_data()
        X = prepare_features(df)
        model = train_model(X)
        scores, flags = score_wallets(model, X)
        result = build_results(df, scores, flags)
        path = save_results(result)

        print(f"\nWallets analysed: {len(result)}")
        print(f"Flagged as anomalous: {int(result['is_anomaly'].sum())}")
        print("\nRisk Level Breakdown:")
        print(result["risk_level"].value_counts().to_string())
        print(f"\nResults saved to: {path}")
        print("\nTop 10 Flagged Wallets:")
        print(result[["rank", "from", "anomaly_score", "risk_level", "is_anomaly"]].head(10).to_string(index=False))
    else:
        # Fall back to running end-to-end pipeline from raw_data
        run_pipeline()


if __name__ == "__main__":
    main()