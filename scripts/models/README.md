# Model Scripts for Groundwater Time Series
This directory contains standalone Python scripts for the benchmark models.

## Overview of Scripts
| Script          | Description                                                                                                 |
|-----------------|-------------------------------------------------------------------------------------------------------------|
| `single.py`     | Single Well models, based on a CNN architecture.                                                            |
| `dynonly.py`    | Global LSTM model using dynamic inputs only.                                                                |
| `dynstat.py`    | Global LSTM-MLP model using dynamic and static inputs.                                                      |
| `evaluation.py` | Computes the error metrics for all models and produces some additional evalutation and plotting of results. |


---

## How to Run

All scripts are standalone and require Python 3 with a few common scientific libraries. Each script uses command-line arguments for input/output paths. Please not that for running the scripts, you need the dataset published on Zenodo:  
[https://doi.org/10.5281/zenodo.1234567](https://doi.org/10.5281/zenodo.1234567)

