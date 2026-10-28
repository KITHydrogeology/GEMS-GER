"""Recalculate GEMS-GER benchmark scores from published medians and dynamic data.

Requires Python, NumPy and pandas. Does not train models or select ensembles.
Example: python evaluation.py --benchmark PATH --dynamic PATH --output PATH
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {"single": "single_median", "global_dynonly": "dynonly_median", "global_dynstat": "dynstat_median"}
METRICS = ["NSE", "R2", "RMSE", "Bias"]
TEST_DATES = pd.date_range("2013-01-07", "2022-12-26", freq="W-MON")
ALL_DATES = pd.date_range("1991-01-07", "2022-12-26", freq="W-MON")


def score(observed, simulated, historical_mean):
    error = simulated - observed
    denominator = float(np.sum((observed - historical_mean)**2))
    return {
        "NSE": 1-float(np.sum(error**2))/denominator if denominator > 0 else np.nan,
        "R2": float(np.corrcoef(observed, simulated)[0, 1]**2)
              if observed.std() > 0 and simulated.std() > 0 else np.nan,
        "RMSE": float(np.sqrt(np.mean(error**2))), "Bias": float(np.mean(error)),
    }


def evaluate(benchmark, dynamic, output=None, tolerance=1e-10, verify=True):
    benchmark, dynamic = Path(benchmark), Path(dynamic)
    expected = {}
    if verify:
        for model in MODELS:
            frame = pd.read_csv(benchmark / f"model_performance_{model}.csv", float_precision="round_trip")
            if list(frame.columns) != ["ID"] + METRICS or frame.ID.duplicated().any():
                raise ValueError("Invalid score schema or duplicate IDs: " + model)
            expected[model] = frame.set_index("ID")
        identifiers = set(expected["single"].index)
        if len(identifiers) != 3207 or any(set(frame.index) != identifiers for frame in expected.values()):
            raise ValueError("Expected the same 3207 wells in all three score files")
    else:
        identifiers = {p.stem for p in dynamic.glob("MW_*.csv")}
        if len(identifiers) != 3207:
            raise ValueError("Publication scoring expects all 3207 wells")
    if {p.stem for p in dynamic.glob("*.csv")} != identifiers:
        raise ValueError("Dynamic CSV well set differs from score files")
    ordered = sorted(identifiers, key=lambda value: int(value.removeprefix("MW_")))
    rows = {model: [] for model in MODELS}
    count = 0
    columns = ["ID", "Date"] + list(MODELS.values())
    chunks = pd.read_csv(benchmark / "median_predictions.csv.gz", chunksize=521, float_precision="round_trip")
    for chunk in chunks:
        if count >= len(ordered) or list(chunk.columns) != columns or len(chunk) != 521:
            raise ValueError("Incomplete or invalid prediction block")
        identifier = ordered[count]
        if not chunk.ID.eq(identifier).all() or chunk.Date.tolist() != TEST_DATES.strftime("%Y-%m-%d").tolist():
            raise ValueError("Invalid well/date order: " + identifier)
        values = chunk[list(MODELS.values())].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite predictions: " + identifier)
        raw = pd.read_csv(dynamic / (identifier + ".csv"), parse_dates=[0], index_col=0,
                          dayfirst=True, decimal=".", sep=",")
        raw.index = pd.to_datetime(raw.index, format="%Y-%m-%d")
        if not raw.index.equals(ALL_DATES) or "GWL" not in raw or "GWL_flag" not in raw:
            raise ValueError("Incomplete original GWL dates/columns: " + identifier)
        observations = raw.loc[raw.index >= "2013-01-01", "GWL"].to_numpy(dtype=np.float64)
        historical_mean = float(raw.loc[raw.index < "2013-01-01", "GWL"].mean())
        if not np.isfinite(observations).all() or not np.isfinite(historical_mean):
            raise ValueError("Nonfinite observations: " + identifier)
        # All 521 weeks are used, including imputed observations; flags are not a filter.
        for model, column in MODELS.items():
            rows[model].append({"ID": identifier, **score(observations, chunk[column].to_numpy(), historical_mean)})
        count += 1
        if count % 500 == 0:
            print(f"Recalculated: {count}/3207 wells", flush=True)
    if count != 3207:
        raise ValueError("Missing prediction blocks")
    report = {}
    for model in MODELS:
        actual = pd.DataFrame(rows[model]).set_index("ID")
        maximum_difference = None
        if verify:
            reference = expected[model].reindex(actual.index)
            np.testing.assert_allclose(actual[METRICS].to_numpy(), reference[METRICS].to_numpy(),
                                       rtol=0, atol=tolerance, equal_nan=True)
            maximum_difference = float((actual[METRICS]-reference[METRICS]).abs().max().max())
        report[model] = {"wells": len(actual), "max_absolute_difference": maximum_difference}
        if output is not None:
            destination = Path(output)
            destination.mkdir(parents=True, exist_ok=True)
            actual.reset_index().to_csv(destination / f"model_performance_{model}.csv",
                                        index=False, float_format="%.17g", lineterminator="\n")
    return report


def combine_models(single, dynonly, dynstat, output):
    """Combine the three model exports into the median-only publication format."""
    import gzip
    import io
    import itertools

    paths = [Path(single), Path(dynonly), Path(dynstat)]
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output directory; existing files are preserved")
    output.mkdir(parents=True)
    readers = [pd.read_csv(p, chunksize=521, float_precision="round_trip") for p in paths]
    destination = output / "median_predictions.csv.gz"
    try:
        with destination.open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as stream:
                    count = 0
                    for chunks in itertools.zip_longest(*readers):
                        if any(c is None for c in chunks):
                            raise ValueError("Model exports have different numbers of wells")
                        merged = chunks[0][["ID", "Date"]].reset_index(drop=True)
                        for chunk, column in zip(chunks, MODELS.values()):
                            if list(chunk.columns) != ["ID", "Date", column] or len(chunk) != 521:
                                raise ValueError("Invalid per-model prediction export")
                            if not chunk[["ID", "Date"]].reset_index(drop=True).equals(merged[["ID", "Date"]]):
                                raise ValueError("Model exports have different well/date alignment")
                            merged[column] = chunk[column].to_numpy()
                        merged.to_csv(stream, index=False, header=count==0, float_format="%.17g", lineterminator="\n")
                        count += 1
        return output
    finally:
        for reader in readers:
            reader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, help="Verify the existing combined Zenodo supplement")
    parser.add_argument("--single", type=Path, help="Median CSV export from single.py")
    parser.add_argument("--dynonly", type=Path, help="Median CSV export from dynonly.py")
    parser.add_argument("--dynstat", type=Path, help="Median CSV export from dynstat.py")
    parser.add_argument("--dynamic", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.benchmark:
        if any([args.single,args.dynonly,args.dynstat]):
            parser.error("Use --benchmark or the three per-model exports, not both")
        print(evaluate(args.benchmark,args.dynamic,args.output),flush=True)
    else:
        if not all([args.single,args.dynonly,args.dynstat,args.output]):
            parser.error("Provide --benchmark or all of --single, --dynonly, --dynstat and --output")
        combined = combine_models(args.single,args.dynonly,args.dynstat,args.output)
        print(evaluate(combined,args.dynamic,combined,verify=False),flush=True)
