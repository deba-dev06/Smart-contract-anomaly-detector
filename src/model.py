from sklearn.ensemble import IsolationForest
from src.config import CONTAMINATION, N_ESTIMATORS, RANDOM_STATE


def train_model(X):
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    model.fit(X)
    return model


def score_wallets(model, X):
    scores = -model.score_samples(X)
    flags = model.predict(X) == -1
    return scores, flags