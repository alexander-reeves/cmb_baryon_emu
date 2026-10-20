"""Lightweight JAX inference for NPZ-exported emulator models."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax.nn import sigmoid


def structured_params_to_matrix(raw_params, param_names=None):
    """Convert structured HDF5 params to the tiled matrix used for training."""
    if param_names is None:
        param_names = list(raw_params.dtype.names)
    return np.column_stack([raw_params[name] for name in param_names]).astype(float)


class JaxNpzModel:
    """JAX-only reader for compact ``.npz`` emulator exports."""

    def __init__(self):
        self.weights = []
        self.hyper_params = []
        self.parameter_names = None
        self.metadata = {}
        self.pk_axes = None

    @classmethod
    def from_package(cls, model_name):
        """Load ``src/cmb_baryon_emu/lightweight_models/{model_name}.npz``."""
        model = cls()
        model.restore(package_model_path(model_name))
        return model

    def restore(self, filename):
        """Load a lightweight model from an ``.npz`` path or path stem."""
        path = Path(filename)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")

        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            self.metadata = metadata
            self.parameter_names = json.loads(str(data["parameter_names_json"]))

            n_dense = int(data["n_dense"])
            self.weights = [
                (jnp.asarray(data[f"weight_{i}"]), jnp.asarray(data[f"bias_{i}"]))
                for i in range(n_dense)
            ]
            self.hyper_params = [
                (jnp.asarray(data[f"alpha_{i}"]), jnp.asarray(data[f"beta_{i}"]))
                for i in range(n_dense - 1)
            ]

            self.scaler_mean_in = jnp.asarray(data["scaler_mean_in"])
            self.scaler_scale_in = jnp.asarray(data["scaler_scale_in"])
            self.scaler_mean_out = jnp.asarray(data["scaler_mean_out"])
            self.scaler_scale_out = jnp.asarray(data["scaler_scale_out"])

            self.log_preprocess = bool(metadata["log_preprocess"])
            self.offset = jnp.asarray(data["offset"]) if "offset" in data else None
            self.rescale_by_as = bool(metadata.get("rescale_by_as", False))
            self.cheby_preprocess = bool(metadata.get("cheby_preprocess", False))

            self.has_pca = bool(metadata["has_pca"])
            if self.has_pca:
                self.pca_components = jnp.asarray(data["pca_components"])
                self.pca_mean = jnp.asarray(data["pca_mean"])
                self.pca_scaler_mean = jnp.asarray(data["pca_scaler_mean"])
                self.pca_scaler_scale = jnp.asarray(data["pca_scaler_scale"])

            # P(k) exports: the output is (n_z, n_k) flattened z-major, on these grids
            self.pk_axes = metadata.get("pk_axes")
            if self.pk_axes is not None:
                self.pk_k = np.asarray(data["pk_k"])
                self.pk_z = np.asarray(data["pk_z"])
                self.pk_shape = tuple(int(n) for n in data["pk_shape"])

            if self.cheby_preprocess:
                raise NotImplementedError("NPZ JAX loader does not yet support Chebyshev exports")

        return self

    def predict_matrix(self, params):
        """Predict spectra from a matrix already in the model's training layout."""
        params = jnp.asarray(params)
        expected = int(self.scaler_mean_in.shape[0])
        if params.ndim != 2 or params.shape[1] != expected:
            raise ValueError(f"expected input shape (n, {expected}), got {params.shape}")

        scaled = (params - self.scaler_mean_in.reshape(1, -1)) / self.scaler_scale_in.reshape(1, -1)
        prediction = self._network(scaled)
        return np.asarray(self._inverse_output_transforms(prediction, params))

    def predict(self, data_dict):
        """Predict from a dict when the model input is one scalar per parameter."""
        if self.parameter_names is None:
            raise ValueError("model has no parameter_names metadata")
        if int(self.scaler_mean_in.shape[0]) != len(self.parameter_names):
            raise ValueError(
                "this model was trained on a tiled input matrix; use predict_matrix() instead"
            )

        missing = set(self.parameter_names) - set(data_dict)
        if missing:
            raise ValueError(f"missing parameters: {sorted(missing)}")

        matrix = np.column_stack([data_dict[name] for name in self.parameter_names]).astype(float)
        return self.predict_matrix(matrix)

    def predict_pk(self, data_dict):
        """Predict a P(k) model as (n, n_z, n_k) on the ``pk_z`` x ``pk_k`` grid.

        A model whose ``pk_axes["ratio_to"]`` is set predicts the ratio to that
        spectrum (``pk_nl_boost`` gives P_NL / P_lin), so multiply by its prediction.
        """
        if self.pk_axes is None:
            raise ValueError("this export has no P(k) axes; use predict()")
        return self.predict(data_dict).reshape(-1, *self.pk_shape)

    def _network(self, x):
        for (weight, bias), (alpha, beta) in zip(self.weights[:-1], self.hyper_params):
            x = jnp.dot(x, weight) + bias
            x = (beta + sigmoid(alpha * x) * (1.0 - beta)) * x
        final_weight, final_bias = self.weights[-1]
        return jnp.dot(x, final_weight) + final_bias

    def _inverse_output_transforms(self, prediction, params):
        prediction = prediction * self.scaler_scale_out.reshape(1, -1) + self.scaler_mean_out.reshape(1, -1)

        if self.has_pca:
            prediction = jnp.dot(prediction, self.pca_components) + self.pca_mean
            prediction = prediction * self.pca_scaler_scale + self.pca_scaler_mean

        if self.log_preprocess:
            prediction = jnp.exp(prediction) + 2 * self.offset

        if self.rescale_by_as:
            raise NotImplementedError("A_s rescaling is not implemented for tiled NPZ inputs")

        return prediction


def package_model_path(model_name):
    """Return the packaged lightweight model path for a spectrum/model name."""
    name = model_name[:-4] if model_name.endswith(".npz") else model_name
    return resources.files("cmb_baryon_emu").joinpath("lightweight_models", f"{name}.npz")
