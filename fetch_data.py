import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
ETHERSCAN_V2_URL = "https://api.etherscan.io/v2/api"

# Desired columns in the final CSV output
TARGET_COLUMNS = [
    "timeStamp",
    "from",
    "to",
    "value",
    "gas",
    "gasUsed",
    "functionName",
    "isError",
    "hash"
]


def clean_function_name(func_str: str) -> str:
    """
    Cleans raw functionName signatures for clear ML feature counts.
    Example: 'swapExactTokensForETH(uint256 amountIn, ...)' -> 'swapExactTokensForETH'
    Empty / NaN string -> 'transfer'
    """
    if not func_str or pd.isna(func_str) or str(func_str).strip() == "":
        return "transfer"
    func_str = str(func_str).strip()
    if "(" in func_str:
        return func_str.split("(")[0].strip()
    return func_str


def fetch_contract_transactions(contract_address: str, days: int = 90) -> pd.DataFrame:
    """
    Fetches normal transactions for a given Ethereum contract address from the last `days` days using Etherscan API V2.
    Handles pagination (sort=desc) and rate-limiting delays.
    """
    if not ETHERSCAN_API_KEY or ETHERSCAN_API_KEY == "your_key_here":
        print("[Error] Invalid or missing ETHERSCAN_API_KEY in .env file. Please add your key.")
        return pd.DataFrame(columns=TARGET_COLUMNS)

    # Calculate cutoff Unix timestamp for the 90-day time window
    cutoff_datetime = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_timestamp = int(cutoff_datetime.timestamp())

    print(f"\n--- Fetching transactions for contract: {contract_address} (Last {days} days since {cutoff_datetime.strftime('%Y-%m-%d')}) ---")
    all_transactions = []
    end_block = 99999999
    page_count = 1
    stop_fetching = False

    while not stop_fetching:
        params = {
            "chainid": "1",
            "module": "account",
            "action": "txlist",
            "address": contract_address,
            "startblock": 0,
            "endblock": end_block,
            "page": 1,
            "offset": 10000,
            "sort": "desc",
            "apikey": ETHERSCAN_API_KEY
        }

        try:
            response = requests.get(ETHERSCAN_V2_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"[Error] HTTP request failed for {contract_address}: {e}")
            break

        status = data.get("status")
        message = data.get("message", "")
        result = data.get("result", [])

        # Check API status
        if status != "1":
            if message == "No transactions found":
                print(f"[Info] No (further) transactions found for {contract_address}.")
            else:
                print(f"[Warning] API returned status='{status}', message='{message}', result='{result}'")
            break

        if not isinstance(result, list) or len(result) == 0:
            print(f"[Info] Reached end of transactions list for {contract_address}.")
            break

        # Filter transactions within the last 90 days window
        valid_batch = []
        for tx in result:
            tx_time = int(tx.get("timeStamp", 0))
            if tx_time >= cutoff_timestamp:
                valid_batch.append(tx)
            else:
                # Encountered transaction older than 90 days cutoff
                stop_fetching = True
                break

        print(f"Fetched page {page_count}: {len(valid_batch)} transactions within 90-day window.")
        all_transactions.extend(valid_batch)

        if stop_fetching:
            break

        # Pagination logic for sort=desc: update end_block to (oldest block in batch - 1)
        if len(result) == 10000:
            last_block_number = int(result[-1]["blockNumber"])
            end_block = last_block_number - 1
            page_count += 1
            # Respect rate limit (5 calls/sec on free tier)
            time.sleep(0.25)
        else:
            break

    if not all_transactions:
        print(f"[Result] 0 transactions collected in the last {days} days for {contract_address}.")
        return pd.DataFrame(columns=TARGET_COLUMNS)

    # Convert list of dicts to pandas DataFrame
    df = pd.DataFrame(all_transactions)

    # Keep only target columns
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[TARGET_COLUMNS].copy()

    # Clean functionName parameter signatures for cleaner feature counts
    df["functionName"] = df["functionName"].apply(clean_function_name)

    # Convert unix seconds timestamp to readable datetime
    df["timeStamp"] = pd.to_datetime(df["timeStamp"].astype(int), unit="s")

    # Convert Wei to ETH (1 ETH = 1e18 Wei)
    df["value"] = df["value"].astype(float) / 1e18

    return df


def save_contract_data(filename: str, df: pd.DataFrame):
    """
    Saves the cleaned DataFrame to raw_data/<filename>.csv
    """
    os.makedirs("raw_data", exist_ok=True)
    if not filename.endswith(".csv"):
        filename = f"{filename}.csv"
    output_filepath = os.path.join("raw_data", filename)
    df.to_csv(output_filepath, index=False)
    print(f"Saved {len(df)} transactions to {output_filepath}")


if __name__ == "__main__":
    # Contracts list with human-readable labels and addresses
    contracts = [
        {"name": "uniswap_v2_router", "address": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"},
        {"name": "usdc_token", "address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"},
        {"name": "usdt_tether", "address": "0xdAC17F958D2ee523a2206206994597C13D831ec7"},
        {"name": "weth_token", "address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"}
    ]

    print("=== Smart Contract Behaviour Anomaly Detector: Data Collection ===")
    for item in contracts:
        if isinstance(item, dict):
            contract_name = item.get("name", "contract")
            contract_address = item.get("address")
        else:
            contract_address = item
            contract_name = item

        df_txs = fetch_contract_transactions(contract_address, days=90)
        if not df_txs.empty:
            save_contract_data(contract_name, df_txs)

    print("\n=== Data Collection Completed ===")
