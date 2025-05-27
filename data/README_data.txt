GEMS-GER / data

This folder contains an anonymized dummy dataset used to demonstrate the four preprocessing steps of the GEMS-GER pipeline.

Files:
- wells_raw.csv                 → Original raw (dummy) data
- wells_filtered.csv            → After removing time series with too many NaNs or long consecutive gaps
- wells_outliers_removed.csv    → After interactive detection and removal of outliers
- wells_levelshifts_removed.csv → After interactive validation of sudden level shifts
- wells_imputed.csv             → Final dataset after block-wise multivariate imputation

Note:
These are **anonymized example data** provided solely for the purpose of illustrating the preprocessing workflow.
They are **not suitable for scientific interpretation**.

Updated: May 2025
