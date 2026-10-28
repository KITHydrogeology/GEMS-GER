# -*- coding: utf-8 -*-
"""Global dynstat LSTM benchmark; original train+validation scaling (1991-2012).

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
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import tensorflow as tf
import keras as ks
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K

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

n_steps_in = 52
HPi_lstm_size = 128
HPi_static_size =  128
HPi_comb_size =  256
HPi_dropout =  0.3
HPi_targetlr =  0.001
HPi_epochs = 20
HPi_batchsize = 512

def make_sequences(data, n_steps_in, n_input):
    #make the data sequential
    #modified after Jason Brownlee and machinelearningmastery.com
    X, Y = list(), list()
    # step over the entire history one time step at a time
    for i in range(len(data)):
        # find the end of this pattern
        end_idx = i + n_steps_in
        # check if we are beyond the dataset
        if end_idx >= len(data):
            break
        # gather input and output parts of the pattern
        seq_x = data[i:end_idx, :n_input]
        seq_y = data[end_idx, n_input:]
        X.append(seq_x)
        Y.append(seq_y)
    return np.array(X), np.array(Y)

class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler implementing linear warmup and cosine decay."""

    def __init__(self, warmup_steps, total_steps, target_lr=0.001, start_lr=0.0, hold=0):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.target_lr = target_lr
        self.start_lr = start_lr
        self.hold = hold
        self.global_step = 0  # Initialize step counter

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "learning_rate"):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
        
        # Compute new learning rate
        new_lr = self.lr_warmup_cosine_decay(self.global_step)
        self.model.optimizer.learning_rate.assign(new_lr)  # Assign new LR
        
        print(f"\nEpoch {epoch}: Learning rate is {float(new_lr)}")
        self.global_step += 1  # Increment step counter

    def lr_warmup_cosine_decay(self, global_step):
        """Computes learning rate with linear warmup and cosine decay."""
        if global_step >= self.total_steps:
            return self.target_lr  # Ensure no overshooting beyond total steps

        # Cosine decay
        learning_rate = 0.5 * self.target_lr * (1 + np.cos(np.pi * (global_step - self.warmup_steps - self.hold) / float(self.total_steps - self.warmup_steps - self.hold)))

        # Linear warmup
        warmup_lr = self.target_lr * (global_step / self.warmup_steps)

        # Apply warmup, hold, and decay logic
        if self.hold > 0:
            learning_rate = np.where(global_step > self.warmup_steps + self.hold, learning_rate, self.target_lr)

        learning_rate = np.where(global_step < self.warmup_steps, warmup_lr, learning_rate)
        return learning_rate

class ClearMemory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        K.clear_session()
        gc.collect()

class DataGenerator(Sequence):
    def __init__(self, X_dyn, X_stat, y, batch_size, shuffle=True):
        self.X_dyn = X_dyn
        self.X_stat = X_stat
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(y))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))
    
    def __getitem__(self, index):
        # Generate batch indices
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Generate data
        X_dyn_batch = self.X_dyn[batch_indices]
        X_stat_batch = self.X_stat[batch_indices]
        y_batch = self.y[batch_indices]

        return (X_dyn_batch, X_stat_batch), y_batch

    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def build_model(X_train, X_train_stat=None):
    model_dyn_in = tf.keras.Input(shape=(n_steps_in, X_train.shape[2]))
    model_stat_in = tf.keras.Input(shape=(X_train_stat.shape[1],))
    model_dyn = tf.keras.layers.LSTM(HPi_lstm_size)(model_dyn_in)
    model_dyn = tf.keras.layers.Dropout(HPi_dropout)(model_dyn)
    model_stat = tf.keras.layers.Dense(HPi_static_size, activation='relu')(model_stat_in)
    model_stat = tf.keras.layers.Dropout(HPi_dropout)(model_stat)
    model_comb = tf.keras.layers.concatenate([model_dyn, model_stat])
    model_comb = tf.keras.layers.Dense(HPi_comb_size, activation='relu')(model_comb)
    model_comb = tf.keras.layers.Dropout(HPi_dropout)(model_comb)
    model_output = tf.keras.layers.Dense(units=1, activation='linear', dtype=tf.float32)(model_comb)
    model = tf.keras.Model(inputs=[model_dyn_in, model_stat_in], outputs=model_output)
    optimizer =  tf.keras.optimizers.Adam(epsilon = 0.0001)
    model.compile(loss='mse', optimizer=optimizer)
    if model.count_params() != 139905 + 128 * X_train_stat.shape[1]:
        raise ValueError('Unexpected global architecture')
    return model

FEATURES = [
    "HYRAS_pr", "HYRAS_tas", "HYRAS_tasmax", "HYRAS_tasmin", "HYRAS_hurs",
    "DWD_evapo_p", "DWD_evapo_r", "DWD_evapo_fao", "DWD_soil_moist",
    "DWD_soil_temp5cm", "ERA5_sro", "ERA5_ssro", "ERA5_sdwe", "ERA5_sm", "ERA5_sf",
]
LENGTH = 52
DATES = pd.date_range("1991-01-07", "2022-12-31", freq="W-MON")
SPLITS = {"train": (52, 887), "validation": (887, 1148), "test": (1148, 1669)}

def load_station(path):
    frame = pd.read_csv(path, parse_dates=[0], index_col=0, dayfirst=True, decimal=".", sep=",")
    frame.index = pd.to_datetime(frame.index)
    if list(frame.columns) != ["GWL"] + FEATURES + ["GWL_flag"]:
        raise ValueError(f"Unerwartete Spalten/Reihenfolge: {path}")
    if not frame.index.equals(DATES):
        raise ValueError(f"Unvollstaendige/doppelte/falsche Wochendaten: {path}")
    flags = frame["GWL_flag"].astype(str).str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError(f"Unerwartete GWL_flag: {path}")
    data = frame[FEATURES + ["GWL"]]
    if not np.isfinite(data.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"Fehlende/nichtendliche Werte: {path}")
    return data, flags.eq("true").to_numpy()

def standardize_station(data):
    """Both fits use 1148 original train+validation weeks; test data never enter the fit."""
    train = data.loc[data.index < "2013-01-01"]
    if len(train) != 1148 or list(data.columns) != FEATURES + ["GWL"]:
        raise ValueError("Falscher Trainingszeitraum oder Zielspaltenposition.")
    scaler_x = StandardScaler().fit(train)
    scaler_y = StandardScaler().fit(train[["GWL"]])
    normalized = scaler_x.transform(data)
    np.testing.assert_array_equal(normalized[:, 15], scaler_y.transform(data[["GWL"]])[:, 0])
    np.testing.assert_allclose(scaler_y.inverse_transform(normalized[:, 15:16])[:, 0],
                               data["GWL"].to_numpy(), rtol=1e-13, atol=1e-12)
    return normalized, scaler_y

class LazyWindowArray:
    """Read-only facade for the exact original, station-major X concatenation."""
    def __init__(self, normalized, split):
        self.normalized = normalized
        self.start, self.stop = SPLITS[split]
        self.per_station = self.stop - self.start
        self.shape = (len(normalized) * self.per_station, LENGTH, len(FEATURES))
        self.dtype = normalized.dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            indices = np.arange(len(self))[indices]
        indices = np.asarray(indices)
        scalar = indices.ndim == 0
        indices = np.atleast_1d(indices)
        if indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("Ungueltiger globaler Sequenzindex.")
        station = indices // self.per_station
        last = self.start + indices % self.per_station
        times = last[:, None] - LENGTH + np.arange(LENGTH)[None, :]
        values = self.normalized[station[:, None], times, :len(FEATURES)]
        return values[0] if scalar else values

def split_arrays(data, split):
    start, stop = SPLITS[split]
    return LazyWindowArray(data["normalized"], split), data["normalized"][:, start:stop, 15].reshape(-1)

def inverse_predictions(normalized_predictions, data, split):
    """Preserve original sklearn inverse_transform dtype/rounding per station."""
    count = SPLITS[split][1] - SPLITS[split][0]
    values = np.asarray(normalized_predictions)
    if values.shape not in [(len(data["ids"]) * count,), (len(data["ids"]) * count, 1)]:
        raise ValueError("Vorhersageform stimmt nicht mit der Messstellenzuordnung ueberein.")
    values = values.reshape(len(data["ids"]), count)
    result = np.empty(values.shape, dtype=np.float64)
    for index, scaler in enumerate(data["scalers_y"]):
        result[index] = scaler.inverse_transform(values[index, :, None])[:, 0]
    if not np.isfinite(result).all():
        raise ValueError("Nichtendliche ruecktransformierte Vorhersagen.")
    return result

DROP_COLUMNS = ["Proj_ID", "Operator", "Depth", "UpFilter", "LoFilter", "ScrLength",
                "Pumping", "PreState", "Easting (EPSG:3035)", "Northing (EPSG:3035)"]
CATEGORIES = ["AquiferMed", "HUEK250_HU", "HUEK250_RT", "HUEK250_CT",
              "HUEK250_DC", "HUEK250_GC", "HUMUS1000_OC"]
ENCODED_NAMES = ["AquiferMed_0", "HUEK250_HU_0", "HUEK250_RT_0", "HUEK250_CT_0",
                 "HUEK250_DC_0", "huek250_GC_0", "HUMUS1000_OC_0"]
NUMERIC_FEATURES = [
    "Elevation", "HYRAUM_HD", "HYRAUM_MHD", "EU_MOHP_DSD_1", "EU_MOHP_DSD_2",
    "EU_MOHP_DSD_3", "EU_MOHP_SD_1", "EU_MOHP_SD_2", "EU_MOHP_SD_3",
    "EU_MOHP_LP_1", "EU_MOHP_LP_2", "EU_MOHP_LP_3", "BUEK1000_RSA", "GWN1000_GR",
    "HUEK250_K", "HYSOG_SG", "SWR_PR", "CLC_90", "CLC_00", "CLC_06", "CLC_12",
    "CLC_18", "MUNDIALIS_LU", "GMK1000_GU", "DTM20_FD", "DTM20_SL", "DTM20_FA",
    "DTM20_AS", "DTM20_GC", "DTM20_PLC", "DTM20_PRC", "DTM20_TPI", "DTM20_TRI",
    "DTM20_CI", "DTM20_MRI",
]

def load_static(path, ids):
    """Keep the reference's column order, label codes and (-1, 1) scaling."""
    raw = pd.read_csv(path, sep=",", index_col=[0])
    required = [c for c in DROP_COLUMNS if c != "Pumping"] + CATEGORIES + ["MW_ID"]
    if raw.columns.duplicated().any() or not set(required).issubset(raw.columns):
        raise ValueError("Statische Pflichtspalten fehlen oder sind doppelt.")
    # The published data omit Pumping; the script drops it despite that. It is
    # excluded if present, and absence cannot change any retained feature.
    frame = raw.drop(columns=[c for c in DROP_COLUMNS if c in raw.columns]).set_index("MW_ID")
    if frame.index.has_duplicates or frame.index.isna().any() or set(frame.index) != set(ids):
        raise ValueError("Statische IDs muessen genau einmal allen dynamischen Messstellen entsprechen.")
    numeric = frame.drop(columns=CATEGORIES)
    if numeric.columns.tolist() != NUMERIC_FEATURES:
        raise ValueError("Unzulaessige/fehlende statische Merkmale oder falsche Spaltenreihenfolge.")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Fehlende/nichtendliche numerische statische Merkmale.")
    for name in CATEGORIES:
        encoder = LabelEncoder()
        codes = encoder.fit_transform(frame[name].to_numpy())
        values = np.concatenate([values, codes[:, None]], axis=1)
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(values)
    normalized = scaler.transform(values)
    if not np.isfinite(normalized).all():
        raise ValueError("Nichtendliche skalierte statische Merkmale.")
    # Select by the actual MW_ID, never the CSV's incidental row position.
    order = frame.index.get_indexer(ids)
    aligned = normalized[order]
    return aligned

class StaticWindowArray:
    """Repeat the correctly aligned station vector lazily for each target week."""
    def __init__(self, values, split):
        self.values = values
        a, b = SPLITS[split]
        self.per_station = b - a
        self.shape = (len(values) * self.per_station, values.shape[1])
        self.dtype = values.dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            indices = np.arange(len(self))[indices]
        indices = np.asarray(indices)
        if indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("Ungueltiger statischer Sequenzindex.")
        return self.values[indices // self.per_station]

def static_arrays(data, split):
    return StaticWindowArray(data["static"], split)

def prepare_global(dynamic, static_path=None):
    files = sorted(Path(dynamic).glob("MW_*.csv"), key=lambda p: p.name)
    if not files:
        raise ValueError("No MW_*.csv inputs found")
    data = {"ids": [p.stem for p in files], "normalized": [], "scalers_y": []}
    for index, path in enumerate(files):
        frame, _ = load_station(path)
        normalized, scaler = standardize_station(frame)
        data["normalized"].append(normalized)
        data["scalers_y"].append(scaler)
        if (index+1) % 500 == 0:
            print(f"Prepared {index+1}/{len(files)} wells", flush=True)
    data["normalized"] = np.asarray(data["normalized"], dtype=np.float64)
    all_ids = pd.read_csv(static_path, index_col=0)["MW_ID"].tolist()
    static = load_static(static_path, all_ids)
    positions = {identifier:index for index,identifier in enumerate(all_ids)}
    data["static"] = static[[positions[identifier] for identifier in data["ids"]]]
    return data

def fit_member(data, seed_row, checkpoint):
    tf.keras.backend.clear_session()
    gc.collect()
    tf.keras.utils.set_random_seed(seed_row["actual_seed"])
    train_x, train_y = split_arrays(data, "train")
    val_x, val_y = split_arrays(data, "validation")
    model = build_model(train_x, static_arrays(data, "train"))
    train_generator = DataGenerator(train_x, static_arrays(data, "train"), train_y, 512, shuffle=True)
    val_generator = DataGenerator(val_x, static_arrays(data, "validation"), val_y, 512, shuffle=False)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1,
            patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(checkpoint), monitor="val_loss", save_best_only=True),
        CustomLearningRateScheduler(3, 20, target_lr=0.001), ClearMemory(),
    ]
    history = model.fit(train_generator, validation_data=val_generator,
        epochs=20, verbose=2, callbacks=callbacks, shuffle=True,
        workers=1, use_multiprocessing=False, max_queue_size=10)
    del model
    gc.collect()
    # As in the original global code, predict with the saved best-val checkpoint.
    model = tf.keras.models.load_model(str(checkpoint))
    predictions = []
    for split in ["validation", "test"]:
        x, y = split_arrays(data, split)
        inference = DataGenerator(x, static_arrays(data, split), y, 32, shuffle=False)
        normalized = model.predict(inference, verbose=0,
            workers=1, use_multiprocessing=False, max_queue_size=10)
        predictions.append(inverse_predictions(normalized, data, split))
    result = np.concatenate(predictions, axis=1)
    del model, history
    tf.keras.backend.clear_session()
    gc.collect()
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic", type=Path, default=Path("data/dynamic"))
    parser.add_argument("--static", type=Path, default=Path("data/static/metadata_static.csv"))
    parser.add_argument("--output", type=Path, default=Path("results_dynstat"))
    parser.add_argument("--members", type=int, default=20)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    args = parser.parse_args()
    seeds = seed_records(args.members, args.master_seed)
    if args.output.exists():
        parser.error("Use a new output directory; every invocation computes a complete ensemble")
    configure_runtime()
    data = prepare_global(args.dynamic, args.static)
    args.output.mkdir(parents=True)
    pd.DataFrame(seeds).to_csv(args.output / "seeds.csv", index=False)
    arrays = []
    for seed_row in seeds:
        folder = args.output / "members" / f"member_{seed_row['member']:04d}"
        folder.mkdir(parents=True)
        values = fit_member(data, seed_row, folder / "model.h5")
        np.save(folder / "predictions.npy", values, allow_pickle=False)
        arrays.append(values)
        print(f"Global member {seed_row['member']}/{args.members} complete", flush=True)
    positions = {identifier: index for index, identifier in enumerate(data["ids"])}
    def matrices(identifier):
        return np.column_stack([a[positions[identifier], 261:] for a in arrays])
    ensemble = args.output / f"ensemble_{args.members:03d}"
    ensemble.mkdir()
    export_medians(ensemble, "dynstat_median", data["ids"], matrices)

if __name__ == "__main__":
    main()
