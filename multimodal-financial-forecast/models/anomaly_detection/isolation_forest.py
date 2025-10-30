from sklearn.ensemble import IsolationForest
import pandas as pd

def detect_anomalies_isolation_forest(df, features):
    model = IsolationForest(n_estimators=100, contamination=0.01)
    df['anomaly_score'] = model.fit_predict(df[features])
    df['is_anomaly'] = df['anomaly_score'] == -1
    return df
