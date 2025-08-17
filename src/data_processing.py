import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(path: str, resample_freq: str = 'H') -> pd.DataFrame:
    """Load and preprocess raw dataset with chunk processing"""
    chunks = pd.read_csv(
        path,
        sep=';',
        parse_dates={'datetime': ['Date', 'Time']},
        infer_datetime_format=True,
        low_memory=False,
        na_values=['?', 'nan'],
        chunksize=10000
    )
    
    df = pd.concat([process_chunk(c) for c in chunks])
    df = df.resample(resample_freq).mean().ffill()
    return df

def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Process data chunks with dtype optimization"""
    chunk['Global_active_power'] = pd.to_numeric(
        chunk['Global_active_power'], errors='coerce'
    )
    return chunk.set_index('datetime')

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features and lag variables"""
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['lag_24h'] = df['Global_active_power'].shift(24)
    return df.dropna()

def scale_data(df: pd.DataFrame) -> tuple:
    """Scale features using MinMaxScaler"""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Global_active_power', 'hour', 'day_of_week', 'month']])
    return pd.DataFrame(scaled, columns=df.columns, index=df.index), scaler
