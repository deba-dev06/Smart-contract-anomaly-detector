from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Candidate locations for feature dataset
_candidate_data_files = [
    ROOT_DIR / "data-collection" / "ml_data_csv" / "ml_features_with_address.csv",
    ROOT_DIR / "data" / "ml_features_with_address.csv",
]
DATA_FILE = next((p for p in _candidate_data_files if p.exists()), _candidate_data_files[0])

RAW_DATA_DIR = ROOT_DIR / "raw_data"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "wallet_anomaly_results.csv"

ADDRESS_COL = "from"
FEATURE_COLUMNS = [
    "transaction_count",
    "total_value",
    "average_value",
    "total_gas",
    "average_gas",
    "average_gas_used",
    "failed_transactions",
    "unique_functions",
    "failure_rate",
    "gas_efficiency",
]
LOG_COLUMNS = [
    "transaction_count",
    "total_value",
    "average_value",
    "total_gas",
    "average_gas",
    "average_gas_used",
    "failed_transactions",
    "unique_functions",
]

CONTAMINATION = 0.05
N_ESTIMATORS = 200
RANDOM_STATE = 42