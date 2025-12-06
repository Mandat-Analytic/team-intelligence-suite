import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def train_and_predict(scores_df, metric_name, future_periods=5):
    """
    Train a Random Forest model on historical scores and predict future trends.
    """
    # Prepare data
    df = scores_df.copy()
    df['Match_Index'] = range(len(df))
    
    X = df[['Match_Index']]
    y = df[metric_name]
    
    # Train model
    # n_estimators=100 for stability, random_state for reproducibility
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predict historical (for trend line)
    trend_historical = model.predict(X)
    
    # Predict future
    last_idx = df['Match_Index'].max()
    future_X = pd.DataFrame({'Match_Index': range(last_idx + 1, last_idx + 1 + future_periods)})
    forecast = model.predict(future_X)
    
    return trend_historical, forecast, future_X['Match_Index'].values

def get_confidence_intervals(scores_df, metric_name):
    """
    Calculate simple confidence intervals based on historical volatility (std dev).
    """
    mean_val = scores_df[metric_name].mean()
    std_val = scores_df[metric_name].std()
    
    lower_bound = mean_val - 1.96 * std_val
    upper_bound = mean_val + 1.96 * std_val
    
    return lower_bound, upper_bound
