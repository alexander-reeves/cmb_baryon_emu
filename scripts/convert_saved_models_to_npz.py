"""Convert TensorFlow saved emulator models to lightweight JAX NPZ files.

The exported files contain only the arrays needed for inference: dense-layer
weights, custom activation parameters, scalers, parameter names, and optional
PCA metadata. They are intended to be committed under
``src/cmb_baryon_emu/lightweight_models/``.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from cmb_baryon_emu.integrated_model import CustomActivation, CustomLoss


ATTRIBUTE_NAMES = [
    "input_scaler",
    "output_scaler",
    "offset",
    "log_preprocess",
    "pca",
    "pca_scaler",
    "parameter_names",
    "rescaling_factor",
    "rescale_by_as",
    "cheby_preprocess",
    "cheby_meta",
    "galaxy_bias_config",
]


def unpack_attributes(path):
    with path.open("rb") as handle:
        values = pickle.load(handle)

    attrs = dict.fromkeys(ATTRIBUTE_NAMES)
    for name, value in zip(ATTRIBUTE_NAMES, values):
        attrs[name] = value

    attrs["log_preprocess"] = bool(attrs["log_preprocess"])
    attrs["rescale_by_as"] = bool(attrs["rescale_by_as"])
    attrs["cheby_preprocess"] = bool(attrs["cheby_preprocess"])
    return attrs


def tf_network_arrays(model_path):
    model = load_model(
        model_path,
        custom_objects={"CustomActivation": CustomActivation, "CustomLoss": CustomLoss},
        compile=False,
    )
    layer_arrays = [layer.get_weights() for layer in model.layers if layer.get_weights()]
    weights = []
    activations = []
    for i, arrays in enumerate(layer_arrays):
        if i % 2 == 0:
            weights.append(arrays)
        else:
            activations.append(arrays)
    return weights, activations


def export_model(model_stem, output_dir, output_name=None):
    model_h5 = model_stem.with_name(model_stem.name + "_model.h5")
    model_pkl = model_stem.with_suffix(".pkl")
    if not model_h5.exists() or not model_pkl.exists():
        raise FileNotFoundError(f"missing {model_h5} or {model_pkl}")

    attrs = unpack_attributes(model_pkl)
    if attrs["cheby_preprocess"]:
        raise NotImplementedError(f"{model_stem.name}: Chebyshev exports are not supported yet")
    if attrs["galaxy_bias_config"] is not None:
        raise NotImplementedError(f"{model_stem.name}: galaxy-bias exports are not supported yet")

    weights, activations = tf_network_arrays(model_h5)
    arrays = {
        "metadata_json": np.array(
            json.dumps(
                {
                    "format": "cmb_baryon_emu_jax_npz_v1",
                    "source_model": model_stem.name,
                    "log_preprocess": attrs["log_preprocess"],
                    "rescale_by_as": attrs["rescale_by_as"],
                    "cheby_preprocess": attrs["cheby_preprocess"],
                    "has_pca": attrs["pca"] is not None,
                }
            )
        ),
        "parameter_names_json": np.array(json.dumps(list(attrs["parameter_names"]))),
        "n_dense": np.array(len(weights), dtype=np.int64),
        "scaler_mean_in": attrs["input_scaler"].mean_.astype(np.float32),
        "scaler_scale_in": attrs["input_scaler"].scale_.astype(np.float32),
        "scaler_mean_out": attrs["output_scaler"].mean_.astype(np.float32),
        "scaler_scale_out": attrs["output_scaler"].scale_.astype(np.float32),
    }

    if attrs["offset"] is not False and attrs["offset"] is not None:
        arrays["offset"] = np.asarray(attrs["offset"], dtype=np.float32)

    for i, (weight, bias) in enumerate(weights):
        arrays[f"weight_{i}"] = np.asarray(weight, dtype=np.float32)
        arrays[f"bias_{i}"] = np.asarray(bias, dtype=np.float32)

    for i, (alpha, beta) in enumerate(activations):
        arrays[f"alpha_{i}"] = np.asarray(alpha, dtype=np.float32)
        arrays[f"beta_{i}"] = np.asarray(beta, dtype=np.float32)

    if attrs["pca"] is not None:
        arrays["pca_components"] = attrs["pca"].components_.astype(np.float32)
        arrays["pca_mean"] = attrs["pca"].mean_.astype(np.float32)
        arrays["pca_scaler_mean"] = attrs["pca_scaler"].mean_.astype(np.float32)
        arrays["pca_scaler_scale"] = attrs["pca_scaler"].scale_.astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_name or model_stem.name}.npz"
    np.savez_compressed(output_path, **arrays)
    return output_path


def discover_model_stems(input_dir, include_temp=False):
    stems = []
    for h5_path in sorted(input_dir.glob("*_model.h5")):
        stem = h5_path.name[: -len("_model.h5")]
        if not include_temp and stem.endswith("_temp"):
            continue
        pkl_path = input_dir / f"{stem}.pkl"
        if pkl_path.exists():
            stems.append(input_dir / stem)
    return stems


def canonical_spectrum_exports(input_dir):
    """Return one best available stem per spectrum, using *_temp only if needed."""
    spectra = [
        "tt_lensed",
        "tt_unlensed",
        "ee_lensed",
        "ee_unlensed",
        "te_lensed",
        "te_unlensed",
    ]
    exports = []
    for spectrum in spectra:
        final_stem = input_dir / spectrum
        temp_stem = input_dir / f"{spectrum}_temp"
        if final_stem.with_name(final_stem.name + "_model.h5").exists() and final_stem.with_suffix(
            ".pkl"
        ).exists():
            exports.append((final_stem, spectrum))
        elif temp_stem.with_name(temp_stem.name + "_model.h5").exists() and temp_stem.with_suffix(
            ".pkl"
        ).exists():
            exports.append((temp_stem, spectrum))
    return exports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("saved_models"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/cmb_baryon_emu/lightweight_models"),
    )
    parser.add_argument("--model", action="append", help="Model stem to export, e.g. tt_lensed")
    parser.add_argument("--include-temp", action="store_true", help="Also export *_temp models")
    parser.add_argument(
        "--canonical-spectra",
        action="store_true",
        help="Export one NPZ per spectrum, promoting *_temp saves when no final save exists",
    )
    args = parser.parse_args()

    if args.canonical_spectra:
        model_exports = canonical_spectrum_exports(args.input_dir)
    elif args.model:
        model_exports = [(args.input_dir / name, None) for name in args.model]
    else:
        model_exports = [
            (model_stem, None)
            for model_stem in discover_model_stems(args.input_dir, include_temp=args.include_temp)
        ]

    if not model_exports:
        raise SystemExit(f"no models found in {args.input_dir}")

    for model_stem, output_name in model_exports:
        output_path = export_model(model_stem, args.output_dir, output_name=output_name)
        size_mb = output_path.stat().st_size / 1024**2
        print(f"wrote {output_path} ({size_mb:.1f} MiB)")


if __name__ == "__main__":
    main()
