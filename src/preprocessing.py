import numpy as np
from sklearn.preprocessing import StandardScaler
from src.config import FEATURE_COLUMNS, LOG_COLUMNS


def prepare_features(df):
    X = df[FEATURE_COLUMNS].astype(float).copy()
    X[LOG_COLUMNS] = np.log1p(X[LOG_COLUMNS].clip(lower=0))
    return StandardScaler().fit_transform(X)