# -*- coding: utf-8 -*-
"""
Created on Tue May 27 15:32:48 2025
@author: Marc

Detects and validates level shifts in groundwater time series.
Prompts user to confirm removal. Saves results and plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
from pathlib import Path
import shutil
from tqdm import tqdm

# === Base directory (two levels above this script) ===
base_dir = Path(__file__).resolve().parents[2]

# === File paths ===
input_csv = base_dir / "data" / "wells_outliers_removed.csv"
output_csv = base_dir / "data" / "wells_levelshifts_removed.csv"

# === Figure directories ===
fig_dir = base_dir / "figures" / "levelshift"
fig_no_shift_dir = base_dir / "figures" / "no_levelshift"
fig_dir.mkdir(parents=True, exist_ok=True)
fig_no_shift_dir.mkdir(parents=True, exist_ok=True)

# === Load data ===
print(f"📂 Loading input: {input_csv}")
df = pd.read_csv(input_csv, index_col=0, parse_dates=True)
start_date = df.index[-1] - pd.DateOffset(years=30)

# === Detect level shifts ===
print("🔍 Detecting level shifts...")
detected_columns = []

for col in tqdm(df.columns, desc="Analyzing columns"):
    series = df[col].dropna()
    if len(series) < 100:
        continue

    algo = rpt.Pelt(model="rbf").fit(series.values.reshape(-1, 1))
    result = algo.predict(pen=100)

    if len(result) <= 1:
        continue

    break_dates = series.index[result[:-1]]
    recent_breaks = break_dates[break_dates > start_date]

    if not recent_breaks.empty:
        # Show plot for user review
        plt.figure(figsize=(12, 6))
        plt.plot(series.index, series, label='Time Series')
        for bp in recent_breaks:
            plt.axvline(bp, color='red', linestyle='--', label='Change Point')
        plt.title(f'Level Shift in {col}')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Save for documentation
        fig_path = fig_dir / f"{col}_levelshift.png"
        plt.figure(figsize=(12, 6))
        plt.plot(series.index, series, label='Time Series')
        for bp in recent_breaks:
            plt.axvline(bp, color='red', linestyle='--')
        plt.title(f'Level Shift in {col}')
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()

        # Prompt user
        while True:
            user_input = input(f"❓ Is this an artificial level shift in {col}? (y/n): ").strip().lower()
            if user_input in ['y', 'n']:
                break

        if user_input == 'n':
            # Move to 'no_levelshift'
            fig_path.rename(fig_no_shift_dir / fig_path.name)
        else:
            detected_columns.append(col)

# === Remove confirmed artificial shift columns ===
print("\n🧹 Removing confirmed artificial level shifts...")
df_filtered = df.drop(columns=detected_columns)

# === Save result ===
df_filtered.to_csv(output_csv)
print(f"\n✅ Final dataset saved to:\n{output_csv}")
print(f"🧾 Removed {len(detected_columns)} columns due to confirmed shifts.")
