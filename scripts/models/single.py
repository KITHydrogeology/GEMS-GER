# -*- coding: utf-8 -*-
"""Single-well CNN benchmark; training-only standardization (1991-2007).

Original model functions/architecture: Tanja Liesch, GEMS-GER.
Standalone 20-member update: documented prefix-stable seeds, original scaling,
per-station inverse transformation and unrounded pointwise median exports.
Complete runs use a new output folder; no automatic resume is performed.
No local project modules, configuration files or external seed files are required.
"""
import os
import sys

# Set before importing TensorFlow, including for direct command-line execution.
_ENV = {"PYTHONHASHSEED":"0", "TF_DETERMINISTIC_OPS":"1", "CUBLAS_WORKSPACE_CONFIG":":4096:8",
        "TF_NUM_INTRAOP_THREADS":"8", "TF_NUM_INTEROP_THREADS":"2", "PYTHONUNBUFFERED":"1", "PYTHONUTF8":"1"}
if __name__ == "__main__" and any(os.environ.get(k) != v for k,v in _ENV.items()):
    environment = os.environ.copy()
    environment.update(_ENV)
    os.execve(sys.executable, [sys.executable]+sys.argv, environment)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import gc
import gzip
import hashlib
import io
import random
from pathlib import Path
import numpy as np
from numpy.random import seed
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras as ks
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten
from keras.models import Model

# Reproducibility and output helpers; no external project modules are required.
POLICY = "GEMS-GER-single-SHA256-v1"
DEFAULT_MASTER_SEED = 20261020

def configure_runtime():
    if tf.__version__ != "2.10.1" or ks.__version__ != "2.10.0":
        raise RuntimeError("Reference runtime: TensorFlow 2.10.1 and Keras 2.10.0")
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()

def export_medians(output, column, ids, matrices):
    """Write one compressed CSV per model, without rounding predictions."""
    path = output / f"median_predictions_{column.removesuffix('_median')}.csv.gz"
    temporary = path.with_suffix(".pending.gz")
    dates = pd.date_range("2013-01-07", "2022-12-26", freq="W-MON").strftime("%Y-%m-%d")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                for index, identifier in enumerate(sorted(ids, key=lambda v: int(v.removeprefix("MW_")))):
                    matrix = matrices(identifier)
                    if matrix.ndim != 2 or matrix.shape[0] != 521 or not np.isfinite(matrix).all():
                        raise ValueError("Wrong prediction shape: " + identifier)
                    frame = pd.DataFrame({"ID": identifier, "Date": dates, column: np.median(matrix, axis=1)})
                    frame.to_csv(stream, index=False, header=index==0, float_format="%.17g", lineterminator="\n")
    os.replace(temporary, path)
    print("Median predictions:", path, flush=True)
    return path

def seed_records(count, master_seed=DEFAULT_MASTER_SEED):
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 100000:
        raise ValueError("Die Mitgliederzahl muss eine ganze Zahl zwischen 1 und 100000 sein.")
    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
        raise ValueError("master_seed muss eine nichtnegative ganze Zahl sein.")
    records, used = [], set()
    for number in range(1, count + 1):
        collision = 0
        while True:
            key = f"{POLICY}:{master_seed}:{number}:{collision}".encode("ascii")
            # Keep 42+ini in NumPy/TensorFlow's positive signed 32-bit range.
            actual = 43 + int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**31 - 43)
            if actual not in used:
                break
            collision += 1
        used.add(actual)
        records.append({"member": number, "actual_seed": actual,
                        "github_ini": actual - 42, "collision_counter": collision})
    return records

FEATURES = [
    "HYRAS_pr", "HYRAS_tas", "HYRAS_tasmax", "HYRAS_tasmin", "HYRAS_hurs",
    "DWD_evapo_p", "DWD_evapo_r", "DWD_evapo_fao", "DWD_soil_moist",
    "DWD_soil_temp5cm", "ERA5_sro", "ERA5_ssro", "ERA5_sdwe", "ERA5_sm", "ERA5_sf",
]
SETUP = {"batch_size": 16, "epochs": 30, "learning_rate": 1e-3,
         "filters": 256, "kernel_size": 3, "dense_size_cnn": 32}
LENGTH = 52
DATES = {"start": "1991-01-07", "validation_start": "2008-01-01",
         "test_start": "2013-01-01", "end": "2022-12-31"}

def split_data(data, seq_length, start, val_start, test_start, end):
    Train = data[(data.index >= start) & (data.index < val_start)]
    
    Val = data[(data.index >= val_start) & (data.index < test_start)]
    Val_ext = pd.concat([Train.iloc[-seq_length:], Val], axis=0)
    
    Test = data[(data.index >= test_start) & (data.index <= end)]
    Test_ext = pd.concat([Val.iloc[-seq_length:], Test], axis=0)
    
    return Train, Val, Val_ext, Test, Test_ext

def scale_data(Train, Val, Val_ext,Test, Test_ext): 
    scaler = StandardScaler().fit(Train)
    
    Train_n = pd.DataFrame(scaler.transform(Train), index=Train.index, columns=Train.columns)   
    Val_n = pd.DataFrame(scaler.transform(Val), index=Val.index, columns=Val.columns)
    Val_ext_n = pd.DataFrame(scaler.transform(Val_ext), index=Val_ext.index, columns=Val_ext.columns)
    Test_n = pd.DataFrame(scaler.transform(Test), index=Test.index, columns=Test.columns) 
    Test_ext_n = pd.DataFrame(scaler.transform(Test_ext), index=Test_ext.index, columns=Test_ext.columns)

    scaler_gwl = StandardScaler().fit(pd.DataFrame(Train.iloc[:,15]))         
       

    return Train_n, Val_n, Val_ext_n, Test_n, Test_ext_n, scaler_gwl

def sequence(data, seq_length, n_input):
    X, Y = list(), list()
    # step over the entire history one time step at a time
    for i in range(len(data)):
        # find the end of this pattern
        end_idx = i + seq_length
        # check if we are beyond the dataset
        if end_idx >= len(data):
            break
        # gather input and output parts of the pattern
        # seq_x, seq_y = data[i:end_idx, 1:], data[end_idx, 0]
        # X.append(seq_x)
        # Y.append(seq_y)
        seq_x = data[i:end_idx, :n_input]
        seq_y = data[end_idx, n_input:]
        X.append(seq_x)
        Y.append(seq_y)
        
    return np.array(X), np.array(Y)

def build_model(ini, seq_length, SETTINGS, inp):   
    #seed
    seed(42+ini)
    tf.random.set_seed(42+ini)
    
#################### Layer Structure    
    inp = Input(shape=(seq_length, inp.shape[2]), name='input')               
    CNN = Conv1D(filters=SETTINGS["filters"], kernel_size=SETTINGS["kernel_size"],
                 padding='same', activation='relu', name='CNN_1')(inp)       
    Pool = MaxPooling1D(pool_size=2, padding='same', name='max_pool')(CNN)
    Flat = Flatten(name='flatten')(Pool)   
    dense = Dense(SETTINGS["dense_size_cnn"], activation='relu', name='dense_gwl')(Flat)    
    out = Dense(1, activation='linear')(dense)
    
#################### Define Model 
    model = Model(inputs=inp, outputs=out)

#################### Optimizer     
    #optimizer = ks.optimizers.Adam(learning_rate=SETTINGS["learning_rate"], epsilon=10E-3, clipnorm=SETTINGS["clip_norm"], clipvalue=SETTINGS["clip_value"])    
    optimizer = ks.optimizers.Adam(learning_rate=SETTINGS["learning_rate"], epsilon=10E-3)    

#################### Compile    
    model.compile(loss='mse', optimizer=optimizer, metrics=[ks.metrics.MeanAbsoluteError()])            
    return model

def load_well(path):
    # Same reader settings as the supplied GitHub script.
    frame = pd.read_csv(path, parse_dates=[0], index_col=0, dayfirst=True, decimal=".", sep=",")
    # The original also calls pd.to_datetime again after reset_index(). Pandas
    # can leave ISO dates as strings when dayfirst=True cannot parse the column.
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "Date"
    if list(frame.columns) != ["GWL"] + FEATURES + ["GWL_flag"]:
        raise ValueError(f"Unerwartete Spalten oder Reihenfolge: {Path(path).name}")
    expected = pd.date_range(DATES["start"], DATES["end"], freq="W-MON")
    if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.equals(expected):
        raise ValueError(f"Unerwartete, fehlende oder doppelte Wochendaten: {Path(path).name}")
    flags = frame["GWL_flag"].astype(str).str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError(f"GWL_flag muss True/False enthalten: {Path(path).name}")
    flags = flags.eq("true")
    data = frame.drop(columns=["GWL_flag"])
    data = data[[col for col in data.columns if col != "GWL"] + ["GWL"]]
    if list(data.columns[:15]) != FEATURES or data.columns[15] != "GWL" or len(data.columns) != 16:
        raise ValueError("GWL muss nach der Umordnung genau Spalte 15 sein.")
    if not np.isfinite(data.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"Fehlende oder nichtendliche Eingangswerte: {Path(path).name}")
    return data, flags

def prepare_data(data, flags):
    parts = split_data(data, LENGTH, *(pd.Timestamp(DATES[key]) for key in DATES))
    train, val, val_ext, test, test_ext = parts
    if [len(part) for part in parts] != [887, 261, 313, 521, 573]:
        raise ValueError("Die Zeitaufteilung weicht vom Benchmark ab.")
    train_n, val_n, val_ext_n, test_n, test_ext_n, scaler_gwl = scale_data(*parts)
    arrays = {}
    for name, frame in [("train", train_n), ("validation", val_ext_n), ("test", test_ext_n)]:
        x, y = sequence(np.asarray(frame), LENGTH, n_input=15)
        if y.shape[1:] != (1,) or x.shape[1:] != (52, 15):
            raise ValueError("Unerwartete Sequenzform.")
        arrays[name] = (x, y)
    if [len(arrays[name][0]) for name in arrays] != [835, 261, 521]:
        raise ValueError("Unerwartete Anzahl von Sequenzen.")
    np.testing.assert_array_equal(train_n["GWL"].to_numpy(), scaler_gwl.transform(train[["GWL"]]).ravel())
    return {"arrays": arrays, "scaler_gwl": scaler_gwl}

def fit_member(prepared, seed_row):
    ks.backend.clear_session()
    gc.collect()
    random.seed(seed_row["actual_seed"])
    ks.utils.set_random_seed(seed_row["actual_seed"])
    model = build_model(seed_row["github_ini"], 52, SETUP, prepared["arrays"]["train"][0])
    if model.count_params() != 224833:
        raise ValueError("Unexpected CNN architecture")
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5,
        verbose=0, mode="auto", baseline=None, restore_best_weights=True)
    history = model.fit(*prepared["arrays"]["train"], validation_data=prepared["arrays"]["validation"],
        epochs=SETUP["epochs"], batch_size=SETUP["batch_size"], shuffle=True, verbose=2, callbacks=[es])
    # Preserve Keras 2.10 semantics: if early stopping does not trigger, use final
    # epoch weights, exactly as the original Single model and the completed run.
    predictions = []
    for split in ["validation", "test"]:
        normalized = model.predict(prepared["arrays"][split][0], verbose=0)
        predictions.append(prepared["scaler_gwl"].inverse_transform(normalized)[:, 0].astype(np.float64))
    result = np.concatenate(predictions)
    del model, history
    ks.backend.clear_session()
    gc.collect()
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic", type=Path, default=Path("data/dynamic"))
    parser.add_argument("--output", type=Path, default=Path("results_single"))
    parser.add_argument("--members", type=int, default=20)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    args = parser.parse_args()
    seeds = seed_records(args.members, args.master_seed)
    files = sorted(args.dynamic.glob("MW_*.csv"), key=lambda p: int(p.stem.removeprefix("MW_")))
    if not files:
        parser.error("No MW_*.csv inputs found")
    if args.output.exists():
        parser.error("Use a new output directory; every invocation computes a complete ensemble")
    configure_runtime()
    args.output.mkdir(parents=True)
    pd.DataFrame(seeds).to_csv(args.output / "seeds.csv", index=False)
    members = args.output / "members"
    members.mkdir()
    for path in files:
        data, flags = load_well(path)
        prepared = prepare_data(data, flags)
        predictions = np.empty((521, args.members), dtype=np.float64)
        for index, seed_row in enumerate(seeds):
            predictions[:, index] = fit_member(prepared, seed_row)[261:]
            print(f"{path.stem}: member {seed_row['member']}/{args.members} complete", flush=True)
        if not np.isfinite(predictions).all():
            raise ValueError("Nonfinite predictions: " + path.stem)
        np.save(members / (path.stem + ".npy"), predictions, allow_pickle=False)
    def matrices(identifier):
        return np.load(members / (identifier + ".npy"), allow_pickle=False)
    ensemble = args.output / f"ensemble_{args.members:03d}"
    ensemble.mkdir()
    export_medians(ensemble, "single_median", [p.stem for p in files], matrices)

if __name__ == "__main__":
    main()
