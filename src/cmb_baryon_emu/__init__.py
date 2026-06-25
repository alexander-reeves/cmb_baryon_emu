"""cmb_baryon_emu: lightweight CMB C_ell + baryonic-feedback emulator package.

Public API
----------
- :func:`predict_cmb_cls` / :func:`load_config` / :func:`cosmo_to_bcemu` from
  :mod:`cmb_baryon_emu.utils` (only need numpy + yaml; BCemu is imported lazily).
- ``IntegratedModel`` — the TensorFlow/Keras emulator wrapper. Imported lazily on
  first access so that ``import cmb_baryon_emu`` does not require TensorFlow for the
  prediction-only workflow.
- ``JaxNpzModel`` — lightweight JAX inference from packaged ``.npz`` exports.
"""

from .utils import cosmo_to_bcemu, load_config, predict_cmb_cls

__all__ = [
    "predict_cmb_cls",
    "load_config",
    "cosmo_to_bcemu",
    "IntegratedModel",
    "JaxNpzModel",
    "structured_params_to_matrix",
]

__version__ = "0.1.0"


def __getattr__(name):
    # Lazy import: avoids importing TensorFlow unless IntegratedModel is requested.
    if name == "IntegratedModel":
        from .integrated_model import IntegratedModel

        return IntegratedModel
    if name == "JaxNpzModel":
        from .jax_emu_model import JaxNpzModel

        return JaxNpzModel
    if name == "structured_params_to_matrix":
        from .jax_emu_model import structured_params_to_matrix

        return structured_params_to_matrix
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
