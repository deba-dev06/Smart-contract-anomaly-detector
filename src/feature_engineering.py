import os
import glob
from pathlib import Path
import pandas as pd


def extract_features_from_transactions(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts behavioural features per wallet ('from' address) from raw transaction records.
    
    Features computed:
    - transaction_count: total interactions sent by wallet
    - total_value: total Ether transferred (ETH)
    - average_value: mean Ether transferred per transaction
    - total_gas: total gas limit allocated
    - average_gas: mean gas limit per transaction
    - average_gas_used: mean gas units consumed
    - failed_transactions: count of reverted/errored transactions
    - unique_functions: count of distinct contract methods invoked
    - failure_rate: proportion of transactions that failed
    - gas_efficiency: ratio of actual gas consumed to allocated gas
    """
    if raw_df.empty:
        return pd.DataFrame()

    # Ensure required columns are present
    df = raw_df.copy()
    if "from" not in df.columns:
        raise ValueError("Raw transactions must contain a 'from' address column.")

    # Standardize data types
    df["value"] = pd.to_numeric(df.get("value", 0), errors="coerce").fillna(0.0)
    df["gas"] = pd.to_numeric(df.get("gas", 0), errors="coerce").fillna(0.0)
    df["gasUsed"] = pd.to_numeric(df.get("gasUsed", 0), errors="coerce").fillna(0.0)
    df["isError"] = df.get("isError", 0).astype(str).str.strip()
    df["functionName"] = df.get("functionName", "transfer").fillna("transfer")

    # Group by wallet address
    grouped = df.groupby("from")
    
    total_tx = grouped.size()
    total_val = grouped["value"].sum()
    avg_val = grouped["value"].mean()
    tot_gas = grouped["gas"].sum()
    avg_gas = grouped["gas"].mean()
    avg_gas_used = grouped["gasUsed"].mean()
    tot_gas_used = grouped["gasUsed"].sum()
    failed_tx = grouped["isError"].apply(lambda s: (s == "1").sum())
    unique_funcs = grouped["functionName"].nunique()

    features_df = pd.DataFrame({
        "from": total_tx.index,
        "transaction_count": total_tx.values,
        "total_value": total_val.values,
        "average_value": avg_val.values,
        "total_gas": tot_gas.values,
        "average_gas": avg_gas.values,
        "average_gas_used": avg_gas_used.values,
        "failed_transactions": failed_tx.values,
        "unique_functions": unique_funcs.values,
    })

    # Derived behavioural ratios
    features_df["failure_rate"] = features_df["failed_transactions"] / features_df["transaction_count"]
    features_df["gas_efficiency"] = (tot_gas_used.values / features_df["total_gas"].replace(0, 1)).clip(0.0, 1.0)

    return features_df.reset_index(drop=True)


def process_raw_data_folder(raw_folder_path: str) -> pd.DataFrame:
    """Reads all CSV files from raw_folder_path and extracts behavioural features."""
    csv_files = glob.glob(os.path.join(raw_folder_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_folder_path}")

    dfs = [pd.read_csv(f) for f in csv_files]
    combined_raw = pd.concat(dfs, ignore_index=True)
    return extract_features_from_transactions(combined_raw)


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent.parent
    raw_path = current_dir / "raw_data"
    print(f"Extracting features from: {raw_path}")
    features = process_raw_data_folder(str(raw_path))
    print(f"Successfully engineered features for {len(features)} unique wallets.")
    print(features.head())

