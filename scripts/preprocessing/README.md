# Preprocessing Scripts for Groundwater Time Series
This directory contains standalone Python scripts for preprocessing groundwater level time series. The goal is to clean, validate, and prepare the data for further analysis or modeling.
Each script is modular, transparent, and can be executed independently from the command line.

## Overview of Scripts
| Script                          | Description                                                                 |
|--------------------------------|-----------------------------------------------------------------------------|
| `01_filter_data_gaps.py`       | Filters time series by year, removes series with high NaN ratios or long gaps. |
| `02_detect_outliers.py`        | Detects outliers using five methods and prompts the user for manual review. |
| `03_detect_level_shifts.py`    | Identifies sudden level shifts and confirms artificial changes via user input. |
| `04_impute_missing_values.py`  | Performs block-wise imputation using multivariate helper data.              |

---

## How to Run

All scripts are standalone and require Python 3 with a few common scientific libraries. Each script uses command-line arguments for input/output paths.

### Example (from terminal or script):
```bash
python 01_filter_data_gaps.py --input ../data/wells_raw.csv --output ../data/wells_filtered.csv
python 02_detect_outliers.py --input ../data/wells_filtered.csv --output ../data/wells_outliers_removed.csv
python 03_detect_level_shifts.py --input ../data/wells_outliers_removed.csv --output ../data/wells_levelshifts_removed.csv
python 04_impute_missing_values.py --input ../data/wells_levelshifts_removed.csv --helper ../data/wells_raw.csv --output ../data/wells_imputed.csv
```

### Output Structure
Figures (e.g., outliers or level shifts) are saved automatically in the `../data/Figures/` folder relative to the output path:

- `Figures/outlier/`
- `Figures/no_outlier/`
- `Figures/levelshift/`
- `Figures/no_levelshift/`

These allow traceable documentation of each decision made during interactive processing.
## Notes
- All scripts use relative paths for portability and GitHub compatibility.
- Dummy input data (`wells_raw.csv`) is anonymized and included for demonstration purposes.
- The workflow does not require a pipeline. Each script can be reviewed or modified individually.

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- ruptures
- tqdm
- statsmodels

Install via:
```bash
pip install -r requirements.txt
```

## Contact
For questions or collaboration, feel free to reach out via GEMS-GER GitHub Issues.
