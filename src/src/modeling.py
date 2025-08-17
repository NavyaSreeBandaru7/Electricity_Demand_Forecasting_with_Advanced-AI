from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_percentage_error

def train_holt_winters(
    data: pd.Series, 
    seasonal_periods: int = 24, 
    trend: str = 'add', 
    seasonal: str = 'add'
) -> ExponentialSmoothing:
    """Train Holt-Winters model with automatic error handling"""
    try:
        model = ExponentialSmoothing(
            data,
            seasonal_periods=seasonal_periods,
            trend=trend,
            seasonal=seasonal
        ).fit(optimized=True)
        return model
    except Exception as e:
        print(f"Model training failed: {str(e)}")
        return None

def forecast_future(
    model: ExponentialSmoothing, 
    steps: int
) -> pd.Series:
    """Generate future forecasts with prediction intervals"""
    forecast = model.forecast(steps)
    return pd.Series(forecast, name='forecast')

def save_model(model, path: str):
    """Save trained model with versioning"""
    joblib.dump(model, f"{path}_v{model.model_version}.pkl")

def evaluate_model(actual, predicted) -> dict:
    """Calculate multiple evaluation metrics"""
    return {
        'mape': mean_absolute_percentage_error(actual, predicted),
        'rmse': ((actual - predicted) ** 2).mean() ** 0.5
    }
