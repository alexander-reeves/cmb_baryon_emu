"""Generate matter power spectrum + CMB lensing training data for cmb_baryon_emu.

Samples cosmological + BCemu2025 baryon parameters on one Latin hypercube and, per
point, runs CAMB once (``cmb_baryon_emu.utils.predict_lss_spectra``) for

  * pk_lin, pk_nl : linear and nonlinear (HMcode-2020) P(k, z), dark-matter-only,
                    (n_z, n_k) in (Mpc/h)^3 on k [h/Mpc] x z grids. The baryon
                    parameters do not enter them.
  * clpp          : C_L^phiphi with BCemu2025 feedback, L = 2 .. lmax.
  * clpp_dmo      : C_L^phiphi without feedback (same CAMB run, stored for checks).

esub entry points: ``main`` (per-index generation) and ``merge`` (collect the
per-index npz files into one HDF5).
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os

import numpy as np
from tqdm.auto import tqdm

mp.set_start_method("fork", force=True)


def set_omp_threads(fallback: int = 4) -> None:
    """Set OMP_NUM_THREADS from Slurm (or a fallback) before importing CAMB."""
    threads = os.getenv("SLURM_CPUS_PER_TASK")
    if threads is None:
        threads = (
            os.getenv("SLURM_CPUS_ON_NODE")
            or os.getenv("SLURM_JOB_CPUS_PER_NODE")
            or os.getenv("SLURM_NPROCS")
        )
    if threads is None:
        threads = str(min(fallback, mp.cpu_count()))
    os.environ["OMP_NUM_THREADS"] = str(int(threads))


set_omp_threads()
print("OMP_NUM_THREADS =", os.environ["OMP_NUM_THREADS"])

from scipy.stats import qmc  # noqa: E402

from cmb_baryon_emu import data_io, emulator, utils  # noqa: E402

LOGGER = logging.getLogger(__name__)

# Per-sample datasets, fixed order. All float32 (~105 kB/sample, ~51 GB for 485k
# samples): train_emulator.py casts to float32 on load anyway, and 7 significant
# digits are far below the emulator targets.
DATASETS = ["pk_lin", "pk_nl", "clpp", "clpp_dmo"]
FLOAT32_DATASETS = set(DATASETS)

# Shared coordinate axes, written once per file.
AXES = ["pk_k", "pk_z", "ell"]

UNITS = {
    "pk_k": {"units": "h/Mpc"},
    "pk_lin": {"units": "(Mpc/h)^3"},
    "pk_nl": {"units": "(Mpc/h)^3"},
    "clpp": {"units": "raw C_L^phiphi"},
    "clpp_dmo": {"units": "raw C_L^phiphi"},
}

# Omega_nu h^2 for the single 0.06 eV neutrino CAMB puts in by default.
OMNUH2 = 0.06 / 93.14


def setup(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samps_per_index", type=int, required=True)
    parser.add_argument(
        "--n_gridpoints",
        type=int,
        required=True,
        help="total LHC points; should equal n_jobs * num_samps_per_index",
    )
    parser.add_argument(
        "--filename_config",
        type=str,
        required=True,
        help="parameter config YAML (priors for cosmo + baryon params)",
    )
    parser.add_argument(
        "--scratch", type=str, required=True, help="output directory for training data"
    )
    parser.add_argument("--lmax", type=int, default=utils.LSS_LMAX)
    parser.add_argument("--lens_potential_accuracy", type=int, default=8)
    parser.add_argument("--q2", type=float, default=0.70)
    parser.add_argument("--nl_model", type=str, default="mead2020")
    parser.add_argument(
        "--omega_m_max", type=float, default=1.0,
        help="skip LHC points with Omega_m above this (no dark energy left)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbosity", type=str, default="warning")
    args = parser.parse_args(args)

    logging.basicConfig(level=getattr(logging, args.verbosity.upper(), logging.WARNING))

    os.makedirs(args.scratch, exist_ok=True)

    config = utils.load_config(args.filename_config)
    parameters = config["parameters"]
    distributions = [p["distribution"] for p in parameters]

    prior_ranges = []
    for p in parameters:
        if p["distribution"] == "flat":
            prior_ranges.append((p["min"], p["max"]))
        elif p["distribution"] == "gaussian":
            kw = p.get("distribution_kwargs", {})
            mu, cov = kw.get("mu"), kw.get("cov")
            if mu is None or cov is None:
                raise ValueError(f"Gaussian param {p['name']} needs mu and cov")
            prior_ranges.append((mu, np.sqrt(cov)))
        else:
            raise ValueError(f"Unknown distribution: {p['distribution']}")

    return parameters, prior_ranges, distributions, config.get("fixed", {}), args


def main(indices, args):
    parameters, prior_ranges, distributions, fixed, args = setup(args)
    param_names = [p["name"] for p in parameters]

    # Every esub job rebuilds the hypercube and takes its own slice, so it must be
    # the same one: np.random.seed does not reach scipy.stats.qmc (SciPy >= 1.15).
    lhd = qmc.LatinHypercube(
        d=len(parameters), rng=np.random.default_rng(args.seed)
    ).random(n=args.n_gridpoints)
    sampled_values = emulator.sample_from_hypercube(lhd, prior_ranges, distributions)

    for index in indices:
        start = index * args.num_samps_per_index
        end = start + args.num_samps_per_index
        sub_samples = sampled_values[start:end]

        spec_lists = {s: [] for s in DATASETS}
        params_list = []
        axis_grids = {}
        bad = []
        n_skipped = 0

        pbar = tqdm(range(sub_samples.shape[0]), desc=f"Index {index}", unit="sample")
        for i in pbar:
            params = sub_samples[i]
            params_dict = {**fixed, **dict(zip(param_names, params))}
            cosmo, bcm = utils.split_cosmo_baryon(params_dict)
            omega_m = (cosmo["omega_b"] + cosmo["omega_cdm"] + OMNUH2) / cosmo["h"] ** 2
            if omega_m > args.omega_m_max:
                n_skipped += 1
                continue
            try:
                out = utils.predict_lss_spectra(
                    cosmo,
                    bcm,
                    lmax=args.lmax,
                    lens_potential_accuracy=args.lens_potential_accuracy,
                    q2=args.q2,
                    nl_model=args.nl_model,
                )
                for axis in AXES:
                    if axis not in axis_grids:
                        axis_grids[axis] = np.asarray(out[axis])
                for s in DATASETS:
                    arr = np.asarray(out[s])
                    spec_lists[s].append(
                        arr.astype(np.float32) if s in FLOAT32_DATASETS else arr
                    )
                params_list.append(params)
            except Exception as e:  # noqa: BLE001 — skip CAMB/BCemu failures
                LOGGER.warning("index %s sample %s failed: %s", index, i, e)
                bad.append(i)
            pbar.set_postfix({"ok": len(params_list), "bad": len(bad), "skip": n_skipped})

        LOGGER.info(
            "index %s: %s ok, %s bad, %s skipped (Omega_m > %g)",
            index, len(params_list), len(bad), n_skipped, args.omega_m_max,
        )

        save_kwargs = {s: np.asarray(spec_lists[s]) for s in DATASETS}
        # tuple() per row: a list of 1-D arrays with a structured dtype would
        # broadcast every value into all fields (see generate_training_data.py).
        save_kwargs["params"] = np.array(
            [tuple(p) for p in params_list],
            dtype=[(name, float) for name in param_names],
        )
        save_kwargs.update(axis_grids)
        np.savez(os.path.join(args.scratch, f"lss_{index}.npz"), **save_kwargs)
        yield index


def merge(indices, args, *, buffer_rows: int = 1000) -> None:
    """Collect per-index ``lss_<idx>.npz`` files into one ``total_data.h5``."""
    *_, args = setup(args)
    data_io.merge_npz(
        [os.path.join(args.scratch, f"lss_{idx}.npz") for idx in indices],
        os.path.join(args.scratch, "total_data.h5"),
        DATASETS,
        AXES,
        attrs=UNITS,
        buffer_rows=buffer_rows,
    )
