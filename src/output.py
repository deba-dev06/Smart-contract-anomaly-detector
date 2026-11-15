from src.config import OUTPUT_DIR, OUTPUT_FILE


def classify_anomaly(score):
    """Classify anomaly score into LOW / MEDIUM / HIGH risk level."""
    if score >= 0.60:
        return "HIGH"
    elif score >= 0.50:
        return "MEDIUM"
    else:
        return "LOW"


def build_results(df, scores, flags):
    result = df.copy()
    result["anomaly_score"] = scores
    result["is_anomaly"] = flags
    result["risk_level"] = result["anomaly_score"].apply(classify_anomaly)
    result = result.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", result.index + 1)
    return result


def save_results(result):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        result.to_csv(OUTPUT_FILE, index=False)
        return OUTPUT_FILE
    except PermissionError:
        # Fallback if target directory is temporarily locked
        fallback_path = Path.cwd() / "wallet_anomaly_results.csv"
        try:
            result.to_csv(fallback_path, index=False)
            return fallback_path
        except Exception:
            return OUTPUT_FILE