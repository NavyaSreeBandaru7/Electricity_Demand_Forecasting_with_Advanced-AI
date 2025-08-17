import openai
import os
from dotenv import load_dotenv

load_dotenv()

class ReportGenerator:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
    
    def generate_forecast_report(
        self, 
        forecast_data: pd.DataFrame, 
        model_metrics: dict
    ) -> str:
        """Generate natural language report using GPT-4"""
        prompt = f"""
        As an energy analyst, create a technical report for electricity demand forecast.
        Data Summary:
        - Forecast Period: {forecast_data.index[0]} to {forecast_data.index[-1]}
        - Peak Demand: {forecast_data['forecast'].max():.2f} kW at {forecast_data.idxmax()[0]}
        - Model Performance: MAPE={model_metrics['mape']*100:.2f}%, RMSE={model_metrics['rmse']:.2f}
        
        Include:
        1. Key forecast statistics
        2. Model performance interpretation
        3. Recommendations for energy management
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message['content']
