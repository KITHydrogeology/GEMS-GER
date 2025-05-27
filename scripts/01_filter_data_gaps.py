# -*- coding: utf-8 -*-
"""
Created on Tue May 27 15:09:38 2025
@author: Marc

This script:
1. Loads a time series dataset from a CSV file.
2. Applies three data quality filters:
   - Only includes data from 1991 to 2022.
   - Removes time series (columns) with too many missing values (NaNs).
   - Removes time series with long consecutive NaN gaps.
3. Saves the filtered result to a new CSV file.
"""

import pandas as pd
from pathlib import Path

# === INPUT/OUTPUT CONFIGURATION ===
base_dir = Path(__file__).resolve().parents[2] 
input_file = base_dir / "data" / "wells_raw.csv"
output_file = base_dir / "data" / "wells_filtered.csv"

# === FILTER PARAMETERS ===
start_year = 1991       # Only use data from this year onward
max_nan_ratio = 0.20    # Max allowed NaN fraction per column (20%)
max_nan_gap = 12        # Max allowed consecutive NaNs in a column

print(f"Loading input data from: {input_file}")
df = pd.read_csv(input_file, index_col=0, parse_dates=True)

# === Filter by start year ===
df_year_filtered = df[(df.index.year >= start_year) & (df.index.year < 2023)]
df_year_filtered = df_year_filtered.dropna(axis=1, how='all')

print(f"Columns after year filter (≥ {start_year}): {df_year_filtered.shape[1]}")

# === Filter by overall NaN ratio ===
nan_ratios = df_year_filtered.isna().mean()
cols_to_keep = nan_ratios[nan_ratios <= max_nan_ratio].index
df_nan_filtered = df_year_filtered[cols_to_keep]

print(f"Columns after NaN ratio ≤ {max_nan_ratio*100:.0f}%: {df_nan_filtered.shape[1]}")

# === Filter by max consecutive NaN gap ===
nan_gaps = {}
for col in df_nan_filtered.columns:
    is_nan = df_nan_filtered[col].isna().astype(int)
    if is_nan.sum() == len(is_nan):
        max_gap = len(is_nan)
    else:
        groups = (is_nan != is_nan.shift()).cumsum()
        max_gap = is_nan.groupby(groups).sum().max()
    nan_gaps[col] = max_gap

nan_gap_df = pd.DataFrame.from_dict(nan_gaps, orient='index', columns=['max_nan_gap'])
cols_to_drop = nan_gap_df[nan_gap_df['max_nan_gap'] > max_nan_gap].index
df_final = df_nan_filtered.drop(columns=cols_to_drop)

# === Result report ===
print("\n Filtering complete:")
print(f"▶ Original columns: {df.shape[1]}")
print(f"▶ After year filter: {df_year_filtered.shape[1]}")
print(f"▶ After NaN ratio filter: {df_nan_filtered.shape[1]}")
print(f"▶ After max NaN gap filter: {df_final.shape[1]}")
print(f"▶ Final average NaN ratio: {df_final.isna().mean().mean() * 100:.2f}%")

# === Save result ===
df_final.to_csv(output_file)
print(f"\nFiltered dataset saved to:\n{output_file}")
