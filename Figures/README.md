# Figures Directory

This directory contains example visualization outputs generated during the **demonstration** of groundwater time series preprocessing steps.

## Structure

- `outlier/`  
  Example plots highlighting detected outliers in groundwater level time series. These illustrate how anomalies can be visually identified and reviewed.

- `no_outlier/`  
  Plots for time series where detected anomalies were reviewed but not marked as outliers.

- `levelshift/`  
  Example plots showing detected sudden level shifts (breakpoints) in the time series data.

- `no_levelshift/`  
  Plots for suspected level shifts that were reviewed but not confirmed as artificial.

## Purpose

These figures serve as illustrative documentation of manual review steps in the preprocessing pipeline demo. They provide insight into the decision-making process on a sample anonymized dataset.

## Notes

- This workflow and figures are **for demonstration only** and do **not** represent the full GEMS-GER benchmark dataset processing used in publications.
- The figures are saved relative to the output CSV file paths.
- The folder and file structure support easy navigation and review of the example preprocessing steps.
