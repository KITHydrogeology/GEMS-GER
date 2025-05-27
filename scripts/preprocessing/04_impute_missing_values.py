# -*- coding: utf-8 -*-
"""
Created on Tue May 27 15:41:03 2025
@author: Marc

This script performs block-wise multivariate imputation on filtered groundwater time series data.

Key Features:
- Processes data in overlapping blocks (6 years with 3-month overlap).
- Uses only helper columns with ≤ 25% missing values in each block.
- Applies IterativeImputer to fill in missing values.
- Saves the final imputed result as a single CSV file.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tqdm import tqdm

# === Base directory relative to script ===
base_dir = Path(__file__).resolve().parents[2]

# === File paths ===
main_file = base_dir / "data" / "wells_levelshifts_removed.csv"
helper_file = base_dir / "data" / "wells_raw.csv"
output_csv = base_dir / "data" / "wells_imputed.csv"

# === Block settings ===
block_len_weeks = 312      # ~6 years
overlap_weeks = 13         # ~3 months
max_nan_ratio_help = 0.25  # Max 25% NaNs allowed in helper columns

# === Load data ===
print(f"📂 Loading main data from: {main_file}")
main_df = pd.read_csv(main_file, index_col=0, parse_dates=True)

print(f"📂 Loading helper data from: {helper_file}")
helper_df = pd.read_csv(helper_file, index_col=0, parse_dates=True)

# === Align index ===
common_index = main_df.index.intersection(helper_df.index)
main_df = main_df.loc[common_index]
helper_df = helper_df.loc[common_index]

# === Only keep helper columns not in main ===
helper_only = helper_df[[col for col in helper_df.columns if col not in main_df.columns]]

# === Prepare blocks ===
time_index = main_df.index
start_indices = list(range(0, len(time_index) - block_len_weeks + 1, block_len_weeks - overlap_weeks))

imputed_blocks = []

print("🔁 Starting block-wise imputation...")
for i, start_idx in enumerate(tqdm(start_indices, desc="Blocks")):
    print(f"▶ Block {i+1}/{len(start_indices)}: weeks {start_idx} to {start_idx + block_len_weeks}")

    end_idx = start_idx + block_len_weeks
    block_range = time_index[start_idx:end_idx]

    block_main = main_df.loc[block_range]
    block_help = helper_only.loc[block_range]

    # === Filter helper columns ===
    valid_helper_cols = block_help.columns[block_help.isna().mean() <= max_nan_ratio_help]
    block_help = block_help[valid_helper_cols]

    # === Combine main and helper ===
    block_data = pd.concat([block_main, block_help], axis=1)

    try:
        imputer = IterativeImputer(
            max_iter=10,
            n_nearest_features=300,
            skip_complete=True,
            random_state=0
        )
        imputed_array = imputer.fit_transform(block_data)
        imputed_block = pd.DataFrame(imputed_array, index=block_data.index, columns=block_data.columns)

        result_block = imputed_block[main_df.columns]
        imputed_blocks.append(result_block)

        print(f"✅ Block {i+1} imputed: {result_block.shape}")
    except Exception as e:
        print(f"⚠️ Error in block {i+1}: {e}")
        continue

# === Combine and deduplicate ===
print("📦 Combining imputed blocks...")
final_df = pd.concat(imputed_blocks)
final_df = final_df[~final_df.index.duplicated(keep='first')]

# === Save result ===
final_df.to_csv(output_csv)
print(f"\n✅ Imputed dataset saved to:\n{output_csv}")
