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
    if stage > 100:
        return 'Over-Exploited'
    if stage >= 90:
        return 'Critical'
    if stage >= 70:
        return 'Semi-Critical'
    return 'Safe'


def simulate_scenario(district: str, extraction_change: float = 0.0, recharge_change: float = 0.0, years_ahead: int = 5) -> pd.DataFrame:
    """
    Run a simple what-if simulation for a district.
    - Learn baseline Stage(%) trend with Linear Regression on Year->Stage.
    - Apply extraction/recharge percentage changes to last observed values.
    - Scale baseline forecast by ratio of new_stage_static / last_stage to reflect change impact.

    Returns columns: Year, PredictedStage, PredictedCategory, Extraction, Recharge
    """
    df = pd.read_csv(DATA_PATH)
    df = _normalize_columns(df)
    df['Year'] = df['Year'].astype(int)
    if 'Stage' not in df.columns and {'Recharge', 'Extraction'}.issubset(df.columns):
        df['Stage'] = (df['Extraction'] / df['Recharge']) * 100.0

    subset = df[df['District'].str.lower() == str(district).lower()].copy()
    subset = subset.sort_values('Year')
    if subset.empty:
        return pd.DataFrame(columns=['Year', 'PredictedStage', 'PredictedCategory', 'Extraction', 'Recharge'])

    last_row = subset.iloc[-1]
    last_year = int(last_row['Year'])
    last_extraction = float(last_row['Extraction'])
    last_recharge = float(last_row['Recharge'])
    last_stage = float(last_row['Stage']) if 'Stage' in last_row else (last_extraction / max(last_recharge, 1e-6) * 100.0)

    # Apply percentage deltas
    new_extraction = last_extraction * (1.0 + float(extraction_change) / 100.0)
    new_recharge = last_recharge * (1.0 + float(recharge_change) / 100.0)
    new_stage_static = (new_extraction / max(new_recharge, 1e-6)) * 100.0

    # Baseline forecast with LR
    if len(subset) < 2:
        years = np.arange(last_year + 1, last_year + years_ahead + 1)
        baseline = np.full_like(years, fill_value=last_stage, dtype=float)
    else:
        X = subset['Year'].to_numpy().reshape(-1, 1)
        y = subset['Stage'].astype(float).to_numpy()
        model = LinearRegression()
        model.fit(X, y)
        years = np.arange(last_year + 1, last_year + years_ahead + 1)
        baseline = model.predict(years.reshape(-1, 1))

    # Scale baseline by impact ratio
    ratio = (new_stage_static / last_stage) if last_stage > 0 else 1.0
    preds = np.maximum(baseline * ratio, 0.0)

    out = pd.DataFrame({
        'Year': years.astype(int),
        'PredictedStage': preds.round(2),
        'Extraction': [round(new_extraction, 2)] * len(years),
        'Recharge': [round(new_recharge, 2)] * len(years),
    })
    out['PredictedCategory'] = out['PredictedStage'].apply(_category_for_stage)
    return out