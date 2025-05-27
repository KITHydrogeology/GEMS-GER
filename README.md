<p align="center">
  <img src="./docs/GEMS.png" width="400" />
</p>
# GEMS-GER

**GEMS-GER** provides code and benchmark models for the publicly available groundwater monitoring dataset in Germany.

## Dataset

The dataset includes long-term groundwater level time series paired with meteorological forcing data and site-specific environmental features.

The full dataset is hosted on Zenodo and can be accessed here:  
[https://doi.org/10.5281/zenodo.1234567](https://doi.org/10.5281/zenodo.1234567)

## Purpose

This repository contains preprocessing scripts and benchmarking models to analyze and predict groundwater levels using machine learning techniques.

## Preprocessing Pipeline

The dataset requires careful cleaning and preparation. We provide four standalone Python scripts demonstrating the main preprocessing steps on an anonymized dummy dataset included in the `/data` folder:

1. **Filter Data Gaps**  
   Removes time series with excessive missing values or long consecutive gaps.

2. **Detect Outliers**  
   Applies multiple statistical and machine learning methods to identify and remove outliers with manual review.

3. **Detect Sudden Level Shifts**  
   Detects abrupt changes in time series levels and allows interactive validation.

4. **Data Imputation**  
   Performs block-wise multivariate imputation using helper variables to fill remaining missing values.

All scripts are modular, command-line executable, and generate figures documenting each interactive cleaning step.

## Machine Learning Models

This repository includes implementations of three benchmark models for groundwater level prediction, designed to serve as transparent baselines for future studies:

1. **Single-Well CNN Models**  
   Individual models trained separately for each monitoring well using only dynamic meteorological input features.  
   - Architecture: Convolutional Neural Network (CNN) with one convolutional layer (256 filters, kernel size 3), max pooling, flattening, a dense layer (32 units), and a single-unit output layer.  
   - Training: Adam optimizer (learning rate 0.001), batch size 16, max 30 epochs with early stopping (patience 5).  
   - Input: Dynamic features only (weekly input window of 52 weeks).

2. **Global LSTM Model with Dynamic Inputs**  
   A single recurrent neural network model trained on all wells simultaneously using dynamic features only.  
   - Architecture: One Long Short-Term Memory (LSTM) layer with 128 units, dropout rate 0.3, followed by output layers.  
   - Training: Batch size 512, max 20 epochs with early stopping (patience 5), learning rate scheduling targeting 0.001.  
   - Input: Dynamic features only, 52-week input window.

3. **Global LSTM-MLP Model with Dynamic and Static Inputs**  
   An enhanced global model combining dynamic and static input data:  
   - Architecture: Two branches—an LSTM branch processing dynamic inputs (as above) and a Multi-Layer Perceptron (MLP) branch processing static features through a dense layer with 128 units. Outputs are concatenated, followed by a dense layer (256 units) and a single-unit output layer.  
   - Training and input settings are similar to the global LSTM model.  
   - Note: Some static features (e.g., geographic coordinates, depth) are excluded due to limited coverage and presumed low relevance. Categorical static features are label-encoded.

**General Notes:**  
- No hyperparameter tuning was performed; architectures were chosen for simplicity and robustness.  
- Models are evaluated on the last 10 years of data (2013–2022) with training on 1991–2007 and validation on 2008–2012 using early stopping.  
- Predictions are based on ensembles of 10 model initializations, with metrics computed on the median prediction.  
- This setup allows comparison of how adding static information and global modeling affect predictive performance compared to individual single-well CNNs.

---

## License

This project is licensed under the [MIT License](LICENSE).  
![MIT License](https://img.shields.io/badge/license-MIT-green.svg)

---

For questions or contributions, please open an issue on this GitHub repository.

---

*Last updated: May 2025*
