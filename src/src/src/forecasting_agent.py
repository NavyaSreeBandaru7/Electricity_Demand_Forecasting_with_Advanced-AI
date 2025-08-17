import numpy as np
from typing import Dict

class ForecastingAgent:
    def __init__(self, model_params: Dict):
        self.params = model_params
        self.performance_history = []
        
    def optimize_hyperparameters(self, data: pd.Series):
        """Automatically tune HW parameters using grid search"""
        best_score = np.inf
        best_params = {}
        
        for trend in ['add', 'mul']:
            for seasonal in ['add', 'mul']:
                model = train_holt_winters(
                    data, 
                    trend=trend, 
                    seasonal=seasonal,
                    seasonal_periods=self.params['seasonal_periods']
                )
                if model:
                    forecast = model.forecast(24)
                    score = self._calculate_score(data[-24:], forecast)
                    if score < best_score:
                        best_score = score
                        best_params = {'trend': trend, 'seasonal': seasonal}
        
        self.params.update(best_params)
        return best_params
    
    def _calculate_score(self, actual, predicted):
        return np.mean(np.abs(actual - predicted) / actual)
    
    def adaptive_forecast(self, data: pd.Series):
        """Auto-retrain model when drift detected"""
        if self._detect_concept_drift(data):
            print("Concept drift detected - retraining model")
            self.optimize_hyperparameters(data)
        return train_holt_winters(data, **self.params)
    
    def _detect_concept_drift(self, data, window=168):
        recent = data[-window:]
        historical = data[-2*window:-window]
        return abs(recent.mean() - historical.mean()) > 0.1 * historical.std()
