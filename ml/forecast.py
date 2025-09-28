import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'groundwater_data.csv')


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if 'Recharge (MCM)' in df.columns:
        rename_map['Recharge (MCM)'] = 'Recharge'
    if 'Extraction (MCM)' in df.columns:
        rename_map['Extraction (MCM)'] = 'Extraction'
    if 'Stage (%)' in df.columns:
        rename_map['Stage (%)'] = 'Stage'
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _category_for_stage(stage: float) -> str:
    # CGWB-style thresholds
    if stage > 100:
        return 'Over-Exploited'
    if stage >= 90:
        return 'Critical'
    if stage >= 70:
        return 'Semi-Critical'
    return 'Safe'


def forecast_stage(district: str, years_ahead: int = 5) -> pd.DataFrame:
    """Forecast Stage (%) for a district for the next N years using Linear Regression.

    Returns a DataFrame with columns: Year, PredictedStage, PredictedCategory
    """
    df = pd.read_csv(DATA_PATH)
    df = _normalize_columns(df)
    df['Year'] = df['Year'].astype(int)
    # If Stage missing but Recharge/Extraction present, compute Stage
    if 'Stage' not in df.columns and {'Recharge', 'Extraction'}.issubset(df.columns):
        df['Stage'] = (df['Extraction'] / df['Recharge']) * 100.0

    subset = df[df['District'].str.lower() == str(district).lower()].copy()
    subset = subset.sort_values('Year')
    if subset.empty:
        return pd.DataFrame(columns=['Year', 'PredictedStage', 'PredictedCategory'])

    # Prepare features
    X = subset['Year'].to_numpy().reshape(-1, 1)
    y = subset['Stage'].astype(float).to_numpy()

    # Fallback if only one point
    if len(subset) < 2:
        last_year = int(subset['Year'].max())
        last_stage = float(subset['Stage'].iloc[-1])
        years = [last_year + i for i in range(1, years_ahead + 1)]
        preds = [last_stage for _ in years]
    else:
        model = LinearRegression()
        model.fit(X, y)
        last_year = int(subset['Year'].max())
        years = np.arange(last_year + 1, last_year + years_ahead + 1)
        preds = model.predict(years.reshape(-1, 1))

    preds = np.maximum(preds, 0.0)  # no negatives
    out = pd.DataFrame({
        'Year': years,
        'PredictedStage': preds.round(2)
    })
    out['PredictedCategory'] = out['PredictedStage'].apply(_category_for_stage)
    return out