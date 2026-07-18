"""Train a per-spectrum CMB + baryon emulator for cmb_baryon_emu.

Trains one IntegratedModel for a single spectrum dataset (e.g. ``TT_lensed``)
from the HDF5 produced by ``generate_training_data.py``. esub entry point:
``main``.

Defaults follow the multiprobe lesson: TT/EE use log preprocessing, TE uses PCA
(TE is signed, so it cannot be log-transformed). Modelled on
multiprobe_framework/scripts/train_emulator.py.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 — registers Zstd decompressor
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cmb_baryon_emu import emulator, integrated_model

LOGGER = logging.getLogger(__name__)

VALID_SPECTRA = [
    "TT_lensed",
    "TT_unlensed",
    "EE_lensed",
    "EE_unlensed",
    "TE_lensed",
    "TE_unlensed",
]


def setup(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data_file", type=str, required=True)
    parser.add_argument(
        "--spectrum", type=str, required=True, choices=VALID_SPECTRA,
        help="which HDF5 spectrum dataset to train on",
    )
    parser.add_argument("--ntrain", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--path_output", type=str, required=True)
    parser.add_argument("--log_preprocess", action="store_true")
    parser.add_argument("--pca_preprocess", action="store_true")
    parser.add_argument("--npca", type=int, default=512)
    parser.add_argument(
        "--high_ell_boost", type=float, default=1.0,
        help="Multiplicative loss weight at the highest multipole relative to the "
        "lowest, ramped smoothly by --high_ell_power (only applies with "
        "--log_preprocess). 1.0 disables reweighting.",
    )
    parser.add_argument(
        "--high_ell_power", type=float, default=4.0,
        help="Exponent controlling how quickly --high_ell_boost ramps on "
        "(higher = boost concentrated closer to ell_max).",
    )
    parser.add_argument("--verbosity", type=str, default="warning")
    args = parser.parse_args(args)

    logging.basicConfig(level=getattr(logging, args.verbosity.upper(), logging.WARNING))
    if args.log_preprocess and args.pca_preprocess:
        raise ValueError("Choose only one of --log_preprocess / --pca_preprocess")
    Path(args.path_output).mkdir(parents=True, exist_ok=True)
    return args


def main(indices, args):
    num_gpus = len(tf.config.experimental.list_physical_devices("GPU"))
    LOGGER.info("Num GPUs Available: %s", num_gpus)

    for index in indices:
        args = setup(args)

        with h5py.File(args.training_data_file, "r") as f:
            n_total = f["params"].shape[0]
            LOGGER.info("total training samples: %s", n_total)
            x_struct = f["params"][: args.ntrain]
            # float32 halves peak memory vs the source float64 datasets; the
            # exported model is float32 throughout anyway (see convert_saved_models_to_npz.py).
            y_train = f[args.spectrum][: args.ntrain, :].astype(np.float32)

        param_names = list(x_struct.dtype.names)
        # structured array -> plain float matrix (n_samples, n_params)
        x_train = np.column_stack([x_struct[n] for n in param_names]).astype(np.float32)

        y_train, x_train = emulator.remove_nan_rows(y_train, x_train)
        y_train, x_train = emulator.remove_zero_rows(y_train, x_train)
        LOGGER.info("after cleaning: x_train %s, y_train %s", x_train.shape, y_train.shape)
        LOGGER.info("param_names: %s", param_names)

        # ---- output preprocessing ----
        offset = False
        if args.log_preprocess:
            LOGGER.info("log preprocessing")
            offset = np.amin(y_train, axis=0)
            offset[offset > 0] = 0
            y_train = np.log(y_train - 2 * offset)

        pca = None
        pca_scaler = None
        rescaling_factor = None
        if args.pca_preprocess:
            LOGGER.info("PCA preprocessing (npca=%s)", args.npca)
            pca_scaler = StandardScaler().fit(y_train)
            normalized = pca_scaler.transform(y_train)
            pca = PCA(n_components=args.npca)
            pca.fit(normalized)
            LOGGER.info("PCA explained variance sum: %.5f", pca.explained_variance_ratio_.sum())
            y_train = pca.transform(normalized)
            rescaling_factor = np.power(pca.explained_variance_, -1)

        if args.log_preprocess and args.high_ell_boost != 1.0:
            # CustomLoss divides squared_errors by element_weights, so a
            # boost > 1 at high ell needs a correspondingly small weight there.
            n_out = y_train.shape[1]
            ramp = np.linspace(0.0, 1.0, n_out) ** args.high_ell_power
            boost = 1.0 + (args.high_ell_boost - 1.0) * ramp
            rescaling_factor = (1.0 / boost).astype(np.float32)
            LOGGER.info(
                "high-ell loss boost enabled: %.1fx at ell_max (power=%.1f)",
                args.high_ell_boost, args.high_ell_power,
            )

        # ---- scalers + model ----
        input_scaler = StandardScaler().fit(x_train)
        output_scaler = StandardScaler().fit(y_train)

        keras_model = integrated_model.create_model(
            input_dim=x_train.shape[1],
            hidden_layers=[512, 512, 512, 512],
            output_dim=y_train.shape[1],
        )

        model = integrated_model.IntegratedModel(
            keras_model,
            input_scaler,
            output_scaler,
            parameter_names=param_names,
            offset=offset,
            log_preprocess=args.log_preprocess,
            pca=pca,
            pca_scaler=pca_scaler,
            rescaling_factor=rescaling_factor,
            temp_file=f"{args.path_output}/{args.model_name}_temp",
        )
        model.train(
            x_train, y_train, epochs=args.epochs, batch_size=1024, validation_split=0.2
        )
        model.save(f"{args.path_output}/{args.model_name}")
        LOGGER.info("Model saved -> %s/%s", args.path_output, args.model_name)
        yield index
