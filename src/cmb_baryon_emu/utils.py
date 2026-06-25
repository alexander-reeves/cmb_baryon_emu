"""High-level prediction API for CMB C_ell with baryonic feedback.

Thin wrapper over the BCemu spectra module (``BCemu.BCemuCAMB``) that takes
user-friendly (CLASS-style) cosmological parameters plus the BCemu2025 baryon
parameters and returns lensed and unlensed CMB power spectra (TT, TE, EE) up to
arbitrary ell.

Example
-------
>>> from cmb_baryon_emu import utils
>>> cosmo = dict(h=0.6736, omega_b=0.02237, omega_cdm=0.1200,
...              ln10As=3.044, n_s=0.9649, tau=0.0544)
>>> bcm = dict(log10Mc=13.1, Theta_co=0.3, mu=1.0, delta=6.0,
...           eta=0.10, deta=0.22, Nstar=0.028, fb=0.0486 / 0.306)
>>> out = utils.predict_cmb_cls(cosmo, bcm, lmax=10000)
>>> out["TT_lensed"].shape, out["ell"][0]
((9999,), 2)
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import yaml

# Canonical parameter names handled by this package.
COSMO_PARAMS = ["omega_cdm", "omega_b", "ln10As", "n_s", "h", "tau"]
BARYON_PARAMS = ["log10Mc", "Theta_co", "mu", "delta", "eta", "deta", "Nstar", "fb"]

DEFAULT_LMAX = 10000
DEFAULT_ELL_MIN = 2

# Maps a (spectrum, kind) request to the BCemu cmb_spectrum() dict key.
_BCEMU_KEY = {
    ("TT", "lensed"): "C_TT",
    ("EE", "lensed"): "C_EE",
    ("TE", "lensed"): "C_TE",
    ("TT", "unlensed"): "C_TT_unlensed",
    ("EE", "unlensed"): "C_EE_unlensed",
    ("TE", "unlensed"): "C_TE_unlensed",
}


def load_config(path):
    """Load a YAML parameter config (same schema as the multiprobe emulator configs)."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def cosmo_to_bcemu(cosmo):
    """Map a CLASS-style cosmology dict to BCemu's ``cosmo_params`` dict.

    ``omega_b``/``omega_cdm`` are physical densities (omega = Omega h^2), so they map
    1:1 onto CAMB's ``ombh2``/``omch2``. ``ln10As`` is converted to ``As``.
    """
    return dict(
        H0=100.0 * cosmo["h"],
        ombh2=cosmo["omega_b"],
        omch2=cosmo["omega_cdm"],
        tau=cosmo["tau"],
        ns=cosmo["n_s"],
        As=np.exp(cosmo["ln10As"]) * 1e-10,
    )


def split_cosmo_baryon(params):
    """Split a flat parameter dict into (cosmo, bcm) sub-dicts by known names."""
    cosmo = {k: params[k] for k in COSMO_PARAMS if k in params}
    bcm = {k: params[k] for k in BARYON_PARAMS if k in params}
    return cosmo, bcm


def predict_cmb_cls(
    cosmo,
    bcm,
    lmax=DEFAULT_LMAX,
    ell_min=DEFAULT_ELL_MIN,
    spectra=("TT", "TE", "EE"),
    kind="both",
    feedback="BCemu2025",
    q2=0.70,
    kmax=50.0,
    verbose=False,
):
    """Predict lensed and/or unlensed CMB C_ell with baryonic feedback.

    Parameters
    ----------
    cosmo : dict
        CLASS-style cosmology: ``h, omega_b, omega_cdm, ln10As, n_s, tau``.
    bcm : dict
        BCemu2025 baryon parameters: ``log10Mc, Theta_co, mu, delta, eta, deta,
        Nstar, fb``.
    lmax : int
        Maximum multipole (default 10000).
    ell_min : int
        Smallest multipole returned; rows ell < ell_min are dropped (default 2).
    spectra : tuple of str
        Which spectra to return, any subset of ``("TT", "TE", "EE")``.
    kind : {"both", "lensed", "unlensed"}
        Which lensing variants to return.
    feedback, q2 : passed to ``BCemu.BCemuCAMB``.
    kmax : float
        Max wavenumber for the matter-power grids [Mpc^-1].
    verbose : bool
        If False, suppress BCemu's stdout chatter (CAMB setup + Limber progress).

    Returns
    -------
    dict
        ``{"ell": ndarray}`` plus one entry per requested spectrum/kind, keyed
        ``"<SPEC>_<lensed|unlensed>"`` (e.g. ``"TT_lensed"``), each an ndarray of
        raw C_ell in muK^2 over ell = ell_min .. lmax.
    """
    import BCemu  # local import: heavy (pulls in CAMB)

    if kind == "both":
        kinds = ("lensed", "unlensed")
    elif kind in ("lensed", "unlensed"):
        kinds = (kind,)
    else:
        raise ValueError(f"kind must be 'both', 'lensed' or 'unlensed', got {kind!r}")

    cm = _maybe_quiet(verbose)
    with cm:
        model = BCemu.BCemuCAMB(
            bcm,
            baryonic_feedback=feedback,
            q2=q2,
            cosmo_params=cosmo_to_bcemu(cosmo),
            lmax=lmax,
            kmax=kmax,
        )
        out = model.cmb_spectrum(lmax=lmax)

    ell = out["ell"]
    keep = ell >= ell_min
    result = {"ell": ell[keep].astype(int)}
    for spec in spectra:
        for k in kinds:
            result[f"{spec}_{k}"] = np.asarray(out[_BCEMU_KEY[(spec, k)]])[keep]
    return result


def _maybe_quiet(verbose):
    """Return a context manager that silences stdout unless ``verbose``."""
    if verbose:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(open(os.devnull, "w"))
