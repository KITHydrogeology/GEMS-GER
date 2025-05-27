# -*- coding: utf-8 -*-
"""
Created on Tue May 27 15:20:20 2025
@author: Marc

Detects outliers in groundwater time series using five methods.
Asks user whether to replace each with NaN. Saves results and plots.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from pathlib import Path

# === BASE DIRECTORY (goes 2 levels up from scripts/preprocessing) ===
base_dir = Path(__file__).resolve().parents[2]

# === FILE PATHS ===
input_file = base_dir / "data" / "wells_filtered.csv"
output_file = base_dir / "data" / "wells_outliers_removed.csv"

# === FIGURE FOLDERS ===
fig_base_path = base_dir / "figures"
fig_outlier_path = fig_base_path / "outlier"
fig_no_outlier_path = fig_base_path / "no_outlier"
fig_outlier_path.mkdir(parents=True, exist_ok=True)
fig_no_outlier_path.mkdir(parents=True, exist_ok=True)

# === LOAD DATA ===
print(f"Loading input: {input_file}")
df = pd.read_csv(input_file, index_col=0, parse_dates=True)

# === DETECTION FUNCTIONS ===
def detect_isolation_forest(df, contamination=0.001):
    df_interp = df.interpolate(method='time').fillna(df.mean())
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_interp), index=df.index, columns=df.columns)
    outliers = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in df.columns:
        series = df_scaled[col].dropna().values.reshape(-1, 1)
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(series)
        pred = model.predict(series)
        outliers.loc[df_scaled[col].dropna().index, col] = (pred == -1)
    return outliers

def detect_lof(df, contamination=0.001, n_neighbors=50):
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
    outliers = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in df.columns:
        series = df_scaled[col].dropna().values.reshape(-1, 1)
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        pred = model.fit_predict(series)
        outliers.loc[df_scaled[col].dropna().index, col] = (pred == -1)
    return outliers

def detect_zscore(df, window, threshold):
    outliers = pd.DataFrame(False, index=df.index, columns=df.columns)
    df_interp = df.interpolate(method='time')
    for col in df.columns:
        roll_mean = df_interp[col].rolling(window=window, center=True).mean()
        roll_std = df_interp[col].rolling(window=window, center=True).std()
        z = (df_interp[col] - roll_mean) / roll_std
        outliers[col] = z.abs() > threshold
    return outliers

def detect_seasonal_decompose(df, threshold=4, period=52):
    outliers = pd.DataFrame(False, index=df.index, columns=df.columns)
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

    for col in df.columns:
        series = df_scaled[col].dropna()
        if len(series) < period * 2:
            continue
        result = seasonal_decompose(series, model='additive', period=period, extrapolate_trend='freq')
        resid = result.resid.dropna()
        resid_thresh = threshold * resid.std()
        mask = (resid.abs() > resid_thresh)
        outliers.loc[mask.index, col] = mask
    return outliers

# === RUN DETECTION ===
print("Running outlier detection...")
outliers_if = detect_isolation_forest(df)
outliers_lof = detect_lof(df)
outliers_zlong = detect_zscore(df, window=26, threshold=3)
outliers_zshort = detect_zscore(df, window=11, threshold=2.65)
outliers_sdc = detect_seasonal_decompose(df)

# === COMBINE RESULTS ===
threshold = 4
df_cleaned = df.copy()

for col in df.columns:
    combined = (
        outliers_if[col].astype(int) +
        outliers_lof[col].astype(int) +
        outliers_zlong[col].astype(int) +
        outliers_zshort[col].astype(int) +
        outliers_sdc[col].astype(int)
    )
    significant_outliers = combined >= threshold

    if significant_outliers.any():
        # === Show figure in Python
        plt.figure(figsize=(7, 4))
        plt.plot(df.index, df[col], label='Original', linewidth=1)
        plt.scatter(df.index[significant_outliers], df[col][significant_outliers],
                    color='red', label='Outliers', zorder=5)
        plt.title(f'Outliers in {col}')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # === Ask user
        while True:
            user_input = input(f"Replace outliers in {col} with NaN? (y/n): ").strip().lower()
            if user_input in ['y', 'n']:
                break

        save_path = fig_outlier_path if user_input == 'y' else fig_no_outlier_path
        fig_save_path = save_path / f"{col}_outliers.png"

        # Save static version
        plt.figure(figsize=(7, 4))
        plt.plot(df.index, df[col], label='Original', linewidth=1)
        plt.scatter(df.index[significant_outliers], df[col][significant_outliers],
                    color='red', label='Outliers', zorder=5)
        plt.title(f'Outliers in {col}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_save_path)
        plt.close()

        if user_input == 'y':
            df_cleaned.loc[significant_outliers, col] = np.nan

# === SAVE CLEANED RESULT ===
df_cleaned.to_csv(output_file)
print(f"\nCleaned dataset saved to:\n{output_file}")
