# Figures Directory

This directory contains example visualization outputs generated during the **demonstration** of groundwater time series preprocessing steps in the GEMS-GER pipeline.

---

## Relation to Data Folder

The corresponding **Data Folder** contains an anonymized dummy dataset used to demonstrate the four preprocessing steps:

- `wells_raw.csv`  
  Original raw (dummy) data.

- `wells_filtered.csv`  
  After removing time series with too many NaNs or long consecutive gaps.

- `wells_outliers_removed.csv`  
  After interactive detection and removal of outliers.

- `wells_levelshifts_removed.csv`  
  After interactive validation of sudden level shifts.

- `wells_imputed.csv`  
  Final dataset after block-wise multivariate imputation.

> **Note:**  
> These are anonymized example data provided solely to illustrate the preprocessing workflow.  
> They are **not suitable for scientific interpretation**.

---

## Figures Folder Structure

- `outlier/`  
  Example plots highlighting detected outliers in groundwater level time series, illustrating anomaly detection and manual review.

- `no_outlier/`  
  Plots of time series where detected anomalies were reviewed and rejected as outliers.

- `levelshift/`  
  Example plots showing detected sudden level shifts (breakpoints) in the time series data.

- `no_levelshift/`  
  Plots of suspected level shifts reviewed but not confirmed as artificial.

---

## Purpose

These figures serve as illustrative documentation of manual review steps in the preprocessing pipeline demonstration. They provide insight into decision-making on a sample anonymized dataset.

---

## Notes

- This workflow and figures are **for demonstration only** and do **not** represent the full GEMS-GER benchmark dataset processing used in publications.
- Figures are saved relative to the output CSV file paths.
- The folder and file structure supports easy navigation and review of the example preprocessing steps.

---

*Updated: May 2025*
