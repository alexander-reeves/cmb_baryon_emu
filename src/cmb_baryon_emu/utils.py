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

# Default fixed k grid for the stored matter power spectra [Mpc^-1]. CAMB's own
# transfer k grid is cosmology dependent (219 points here, 215-222 across the
# training prior), so P(k) has to be resampled onto a shared grid to be stackable
# across samples. 300 log-spaced points is ~53/decade, denser than CAMB's native
# ~32/decade, so the spline resampling is lossless at the 1e-5 level.
DEFAULT_PK_KMIN = 1e-4
DEFAULT_PK_KMAX = 50.0
DEFAULT_PK_NK = 300

# Default grids for predict_lss_spectra. k is in h/Mpc, log-spaced piecewise: 20 per
# decade on super-horizon scales, 100 per decade through the BAO, 40 per decade in
# the nonlinear regime (329 points). 1e-5 h/Mpc reaches CosmoPower's k_min of
# 1e-5 Mpc^-1 for any h < 1.
LSS_K_H = np.concatenate([
    np.logspace(-5, -2, 60, endpoint=False),
    np.logspace(-2, 0, 200, endpoint=False),
    np.geomspace(1.0, 50.0, 69),
])
LSS_Z = np.round(np.linspace(0.0, 3.0, 31), 10)
LSS_LMAX = 3000

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
    return_ckk=False,
    pk_k=None,
    pk_z=None,
    pk_z_stride=2,
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
    return_ckk : bool
        If True, also return the CMB lensing convergence spectrum ``"clkk"`` on
        the same ell grid. Free: BCemu already computes it for the lensed C_ell.
    pk_k : array, True or None
        If not None, also return matter power spectra. ``True`` uses the default
        grid ``geomspace(DEFAULT_PK_KMIN, DEFAULT_PK_KMAX, DEFAULT_PK_NK)``;
        an array is used as given [Mpc^-1]. Free: the P(k) splines are already
        built by ``BCemuCAMB.__init__``, so this is spline evaluation only.
    pk_z : array or None
        Redshifts for the P(k) output. None takes every ``pk_z_stride``-th node of
        BCemu's own nonlinear redshift grid (z = 0 .. 5), so no z interpolation
        happens at all -- only k is resampled.
    pk_z_stride : int
        Subsampling stride for the default ``pk_z`` (2 -> 21 redshifts).

    Returns
    -------
    dict
        ``{"ell": ndarray}`` plus one entry per requested spectrum/kind, keyed
        ``"<SPEC>_<lensed|unlensed>"`` (e.g. ``"TT_lensed"``), each an ndarray of
        raw C_ell in muK^2 over ell = ell_min .. lmax.

        With ``return_ckk``: ``"clkk"``, C_L^kappakappa on the same ell grid.

        With ``pk_k``: ``"pk_k"`` [Mpc^-1], ``"pk_z"``, and three (n_z, n_k)
        spectra in Mpc^3 -- ``"pk_lin"`` (linear, dark-matter-only), ``"pk_nl"``
        (nonlinear DMO, ``mead2020``) and ``"pk_dmb"`` (= pk_nl * S, the only one
        carrying the baryon parameters). BCemu2025's S is defined for z <= 3.04
        and 0.034-12.5 h/Mpc; outside that range the default ``k_extrap=None``
        policy gives S = 1, so ``pk_dmb == pk_nl`` there.
    """
    import BCemu  # local import: heavy (pulls in CAMB)

    if kind == "both":
        kinds = ("lensed", "unlensed")
    elif kind in ("lensed", "unlensed"):
        kinds = (kind,)
    else:
        raise ValueError(f"kind must be 'both', 'lensed' or 'unlensed', got {kind!r}")

    if pk_k is True:
        pk_k = np.geomspace(DEFAULT_PK_KMIN, DEFAULT_PK_KMAX, DEFAULT_PK_NK)

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
        if pk_k is not None:
            # model._z_nl is the fixed grid CAMB actually solved the nonlinear
            # P(k) on; using its nodes means the z axis is never interpolated.
            z_grid = model._z_nl[::pk_z_stride] if pk_z is None else np.asarray(pk_z)
            pk = model.matter_spectrum(np.asarray(pk_k), z_grid)

    ell = out["ell"]
    keep = ell >= ell_min
    result = {"ell": ell[keep].astype(int)}
    for spec in spectra:
        for k in kinds:
            result[f"{spec}_{k}"] = np.asarray(out[_BCEMU_KEY[(spec, k)]])[keep]
    if return_ckk:
        result["clkk"] = np.asarray(out["C_kappakappa"])[keep]
    if pk_k is not None:
        result["pk_k"] = pk["k"]
        result["pk_z"] = pk["z"]
        result["pk_lin"] = pk["P_L_dmo"]
        result["pk_nl"] = pk["P_NL_dmo"]
        result["pk_dmb"] = pk["P_dmb"]
    return result


def predict_lss_spectra(
    cosmo,
    bcm,
    k_h=None,
    z=None,
    lmax=LSS_LMAX,
    lens_potential_accuracy=8,
    q2=0.70,
    nl_model="mead2020",
    verbose=False,
):
    """Matter power spectra and the baryonic CMB lensing spectrum from one CAMB run.

    One CAMB call (lensing to ``lmax`` plus matter power at ``z``, HMcode
    ``nl_model``) gives the dark-matter-only linear and nonlinear P(k) and the
    dark-matter-only C_L^phiphi. BCemu2025 feedback enters C_L^phiphi only, as a
    Limber correction from P_NL * (S - 1) (see ``_baryon_dclpp``); the bcm
    parameters never reach CAMB, so they do not affect the P(k).

    Parameters
    ----------
    cosmo : dict
        CLASS-style cosmology: ``h, omega_b, omega_cdm, ln10As, n_s, tau``.
    bcm : dict
        BCemu2025 parameters: ``log10Mc, Theta_co, mu, delta, eta, deta, Nstar, fb``.
    k_h : array or None
        Output wavenumbers [h/Mpc]; default ``LSS_K_H`` (1e-5 .. 50, 329 points).
    z : array or None
        Output redshifts; default ``LSS_Z`` (0 .. 3 in steps of 0.1).
    lmax : int
        Largest lensing multipole L (default 3000).
    lens_potential_accuracy : int
        Passed to CAMB's ``set_for_lmax``. 8 keeps C_L^phiphi within 0.08 % of a
        converged (lmax 4500, accuracy 16) run up to L = 3000; 5 gives 0.15 %.
    q2, nl_model : passed to BCemu2025 / CAMB's HMcode.

    Returns
    -------
    dict
        ``"pk_k"`` [h/Mpc], ``"pk_z"``, ``"pk_lin"`` and ``"pk_nl"`` [(Mpc/h)^3,
        shape (n_z, n_k), dark-matter-only], ``"ell"`` (L = 2 .. lmax), ``"clpp"``
        (raw C_L^phiphi with baryons) and ``"clpp_dmo"`` (without).
    """
    import camb
    from camb import model as camb_model

    k_h = LSS_K_H if k_h is None else np.asarray(k_h, dtype=float)
    z = LSS_Z if z is None else np.asarray(z, dtype=float)
    p = cosmo_to_bcemu(cosmo)
    h = p["H0"] / 100.0
    z_emu_max = float(_bcemu2025().z_grid[-1])

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=p["H0"], ombh2=p["ombh2"], omch2=p["omch2"], tau=p["tau"])
    pars.InitPower.set_params(As=p["As"], ns=p["ns"])
    pars.set_for_lmax(lmax, lens_potential_accuracy=lens_potential_accuracy)
    # P_NL is also needed out to BCemu2025's last redshift node for the baryon
    # correction, which can lie beyond the output grid.
    z_camb = np.unique(np.concatenate([z, [z_emu_max]]))
    pars.set_matter_power(redshifts=list(z_camb), kmax=1.2 * k_h[-1] * h, silent=True)
    pars.NonLinear = camb_model.NonLinear_both
    pars.NonLinearModel.set_params(nl_model)
    with _maybe_quiet(verbose):
        res = camb.get_results(pars)

    pk = {}
    for name, nonlinear in (("pk_lin", False), ("pk_nl", True)):
        interp = res.get_matter_power_interpolator(
            nonlinear=nonlinear, hubble_units=True, k_hunit=True
        )
        pk[name] = _pk_on_grid(interp, z, k_h)

    ell = np.arange(2, lmax + 1)
    clpp_dmo = res.get_lens_potential_cls(lmax=lmax, raw_cl=True)[2:, 0]
    dclpp = _baryon_dclpp(res, pars, bcm, ell, h, q2, z_emu_max)
    return {
        "pk_k": k_h,
        "pk_z": z,
        "pk_lin": pk["pk_lin"],
        "pk_nl": pk["pk_nl"],
        "ell": ell,
        "clpp": clpp_dmo + dclpp,
        "clpp_dmo": clpp_dmo,
    }


def baryon_suppression(bcm, k_mpc, z, h, q2=0.70):
    """BCemu2025 suppression S(k, z) = P_dmb / P_dmo on a (z, k) grid, k in Mpc^-1.

    Linear in z between the emulator's redshift nodes (as ``BCemu2025.get_boost``
    interpolates) and cubic in log k. Outside the emulator's range: S = 1 below its
    k_min (0.034 h/Mpc, where |1 - S| < 3e-4) and above its last redshift (3.04);
    above its k_max (12.5 h/Mpc) S is held at the edge value, since S = 1 there
    would put a 15-35 % step into P_dmb.
    """
    from scipy.interpolate import CubicSpline

    emu = _bcemu2025()
    zn = np.asarray(emu.z_grid, dtype=float)
    lkn = np.log(np.asarray(emu.k, dtype=float))
    s_nodes = np.array([np.asarray(emu.get_boost(bcm, float(zi), q2)[1]) for zi in zn])

    lk = np.log(np.atleast_1d(k_mpc) / h)
    s_k = CubicSpline(lkn, s_nodes, axis=1)(np.clip(lk, lkn[0], lkn[-1]))
    s_k[:, lk < lkn[0]] = 1.0

    z = np.atleast_1d(z).astype(float)
    zc = np.clip(z, zn[0], zn[-1])
    i1 = np.clip(np.searchsorted(zn, zc), 1, len(zn) - 1)
    w = ((zc - zn[i1 - 1]) / (zn[i1] - zn[i1 - 1]))[:, None]
    s = (1.0 - w) * s_k[i1 - 1] + w * s_k[i1]
    s[z > zn[-1]] = 1.0
    return s


def _baryon_dclpp(res, pars, bcm, ell, h, q2, z_max):
    """Baryonic change in C_L^phiphi: Limber integral of P_NL,dmo (S - 1) over z <= z_max.

    CAMB itself uses Limber for C_L^phiphi at L >= 100 (``limber_phi_lmin``), and
    S - 1 is below 3e-4 at the k where non-Limber terms matter. Against a second
    CAMB run fed P_NL * S through ``ExternalNonLinearRatio`` this agrees to
    <= 4.4e-4 of C_L for L <= 3000, even for strong feedback (-7.7 % at L = 3000).
    """
    c_kms = 2.99792458e5
    pk_nl = res.get_matter_power_interpolator(
        nonlinear=True, hubble_units=False, k_hunit=False, extrap_kmax=1e4
    )
    zs = np.concatenate([np.geomspace(1e-3, 0.1, 60), np.linspace(0.1, z_max, 300)[1:]])
    # S - 1 vanishes below BCemu2025's k_min of 0.034 h/Mpc.
    kg = np.geomspace(0.03 * h, 1e3, 700)
    f = pk_nl.P(zs, kg) * (baryon_suppression(bcm, kg, zs, h, q2) - 1.0)

    chi = res.comoving_radial_distance(zs)
    chistar = res.conformal_time(0) - res.tau_maxvis
    omega_m = (pars.ombh2 + pars.omch2 + pars.omnuh2) / h**2
    w_kappa = 1.5 * omega_m * (pars.H0 / c_kms) ** 2 * chi * (chistar - chi) / chistar * (1 + zs)

    lkg = np.log(kg)
    integrand = np.empty((len(ell), len(zs)))
    for j in range(len(zs)):
        f_at_l = np.interp(np.log((ell + 0.5) / chi[j]), lkg, f[j], left=0.0, right=f[j, -1])
        integrand[:, j] = f_at_l * (w_kappa[j] / chi[j]) ** 2
    dckk = np.trapezoid(integrand, chi, axis=1)
    return 4.0 * dckk / (ell * (ell + 1.0)) ** 2


def _pk_on_grid(interp, z, k_h):
    """CAMB ``interp.P`` on the (z, k) grid, extended below CAMB's k_min as a power law.

    CAMB's smallest k is ~7e-6 Mpc^-1, which is above 1e-5 h/Mpc when h < 0.7.
    Those scales are far outside the horizon, where the local slope (~n_s) is constant.
    """
    inside = k_h >= interp.kmin
    out = np.empty((len(z), len(k_h)))
    out[:, inside] = interp.P(z, k_h[inside])
    if not inside.all():
        k_ref = np.array([interp.kmin, 1.1 * interp.kmin])
        p_ref = interp.P(z, k_ref)
        slope = np.log(p_ref[:, 1] / p_ref[:, 0]) / np.log(1.1)
        out[:, ~inside] = p_ref[:, :1] * (k_h[~inside] / interp.kmin) ** slope[:, None]
    return out


_BCEMU2025 = None


def _bcemu2025():
    """The BCemu2025 emulator, loaded once per process (each load takes ~1.4 s)."""
    global _BCEMU2025
    if _BCEMU2025 is None:
        from BCemu.BaryonEffectsEmulator import BCemu2025

        with _maybe_quiet(False):
            _BCEMU2025 = BCemu2025(backend="numpy")
    return _BCEMU2025


def _maybe_quiet(verbose):
    """Return a context manager that silences stdout unless ``verbose``."""
    if verbose:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(open(os.devnull, "w"))
