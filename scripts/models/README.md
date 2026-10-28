# Model Scripts for Groundwater Time Series

This directory contains the three GEMS-GER benchmark models and the score evaluation script.

| Script | Description |
|---|---|
| `single.py` | Single-well CNN using dynamic inputs. |
| `dynonly.py` | Global LSTM using dynamic inputs. |
| `dynstat.py` | Global LSTM-MLP using dynamic and static inputs. |
| `evaluation.py` | Combines model median exports and recalculates or verifies the per-well scores. |

## Run the models

Use the GEMS-GER input data. The default paths are `data/dynamic/` for dynamic CSVs and, for `dynstat.py`, `data/static/metadata_static.csv`. Output paths can be set with `--output`.

```text
python scripts/models/single.py --dynamic data/dynamic --output results_single --members 20
python scripts/models/dynonly.py --dynamic data/dynamic --output results_dynonly --members 20
python scripts/models/dynstat.py --dynamic data/dynamic --static data/static/metadata_static.csv --output results_dynstat --members 20
```

Each invocation computes a complete ensemble in a **new output directory**. Existing output directories are rejected; automatic resume is not included. For a rerun or an ensemble extension, choose another `--output` path.

The default ensemble size is 20. Seeds are generated from master seed `20261020` by the same deterministic policy embedded in all three scripts and recorded in the output's `seeds.csv`. Increasing `--members` to 21 preserves the first 20 seeds; the complete ensemble is recalculated.

The scripts retain the original model architectures and training settings. Single-well predictors are standardized on 1991–2007; global predictors and groundwater levels are standardized on 1991–2012. Each model writes a median-prediction CSV under its `ensemble_020` output directory.

The reference training environment is Python 3.9.21, TensorFlow 2.10.1, Keras 2.10.0, NumPy 1.26.4, pandas 2.3.0 and scikit-learn 1.6.1. The scripts use the GPU when TensorFlow detects one. Exact numerical agreement across different hardware or software environments is not guaranteed.

## Changes from the original implementation

Model architectures, hyperparameters, split dates and scaling periods are retained. The implementation also makes these explicit technical changes:

- All three models use the same 20-member seed policy. Python, NumPy and TensorFlow random generators are seeded, deterministic operations are enabled, and the session is reset before each member.
- Input filenames have a fixed order. Global dynamic windows and repeated static features are created on demand to reduce memory use; their values and temporal alignment follow the original sequence construction.
- Input columns, dates and finite values are checked. The excluded `Pumping` column may be absent from the static input.
- Predictions are inverse-transformed per well and exported without the global scripts' former four-decimal rounding. The median is formed at each test week before score calculation.

The model scripts save member predictions and the median exports. `evaluation.py` reproduces the four benchmark metrics; the former plotting and GIS sections are omitted. The scripts contain no file-hash registry, run-lock class or cache/resume framework. SHA-256 is used only by the shared seed policy to derive reproducible seeds.

## Reproduce the scores

To combine the three per-model median files and calculate the per-well scores:

```text
python scripts/models/evaluation.py --single results_single/ensemble_020/median_predictions_single.csv.gz --dynonly results_dynonly/ensemble_020/median_predictions_dynonly.csv.gz --dynstat results_dynstat/ensemble_020/median_predictions_dynstat.csv.gz --dynamic data/dynamic --output benchmark_update
```

To verify score files supplied with the Zenodo supplement:

```text
python scripts/models/evaluation.py --benchmark Zenodo_Update --dynamic data/dynamic
```

Score reproduction alone does not require TensorFlow; Python, NumPy and pandas are sufficient.
