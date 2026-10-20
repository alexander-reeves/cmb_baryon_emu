"""Generate CMB + baryon training data for the cmb_baryon_emu emulator.

Samples cosmological + BCemu2025 baryon parameters on a Latin hypercube and
computes lensed & unlensed TT/TE/EE C_ell (via cmb_baryon_emu.utils ->
BCemu.BCemuCAMB) for each point. esub entry points: ``main`` (per-index
generation) and ``merge`` (collect per-index npz into one HDF5).

Each sample also stores the CMB lensing spectrum C_L^kk and the matter power
spectra (linear DMO, nonlinear DMO, and baryon-suppressed) out to k = 50 Mpc^-1.
Both come from the CAMB solves BCemuCAMB already runs for the C_ell, so they add
no compute -- see ``utils.predict_cmb_cls``.

Modelled on multiprobe_framework/scripts/generate_emulator_training_data.py.
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

# Output spectra (HDF5 dataset names), fixed order.
SPECTRA = [
    "TT_lensed",
    "TT_unlensed",
    "EE_lensed",
    "EE_unlensed",
    "TE_lensed",
    "TE_unlensed",
]

# CMB lensing convergence spectrum, on the same ell grid as SPECTRA.
CLKK = ["clkk"]

# Matter power spectra, shape (n_z, n_k) per sample. pk_lin/pk_nl are
# dark-matter-only (cosmology only); only pk_dmb carries the baryon parameters.
PK_SPECTRA = ["pk_lin", "pk_nl", "pk_dmb"]

# Shared coordinate axes, written once per HDF5 file rather than per sample.
AXES = ["ell", "pk_k", "pk_z"]

# Stored as float32: these are the bulk of the per-sample payload and
# train_emulator.py casts to float32 on load anyway.
FLOAT32_DATASETS = set(PK_SPECTRA)


def dataset_names(args):
    """Per-sample dataset names produced for this run, in fixed order."""
    names = list(SPECTRA)
    if not args.no_clkk:
        names += CLKK
    if not args.no_pk:
        names += PK_SPECTRA
    return names


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
    parser.add_argument("--lmax", type=int, default=10000)
    parser.add_argument("--ell_min", type=int, default=2)
    parser.add_argument("--kmax", type=float, default=50.0)
    parser.add_argument("--feedback", type=str, default="BCemu2025")
    parser.add_argument("--q2", type=float, default=0.70)
    parser.add_argument(
        "--pk_nk", type=int, default=utils.DEFAULT_PK_NK,
        help="number of log-spaced k points for the stored P(k)",
    )
    parser.add_argument("--pk_kmin", type=float, default=utils.DEFAULT_PK_KMIN)
    parser.add_argument("--pk_kmax", type=float, default=utils.DEFAULT_PK_KMAX)
    parser.add_argument(
        "--pk_z_stride", type=int, default=2,
        help="stride over BCemu's nonlinear z nodes (2 -> 21 redshifts, z = 0..5)",
    )
    parser.add_argument(
        "--no_pk", action="store_true", help="skip the matter power spectra"
    )
    parser.add_argument(
        "--no_clkk", action="store_true", help="skip the CMB lensing spectrum"
    )
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

    return parameters, prior_ranges, distributions, args


def main(indices, args):
    parameters, prior_ranges, distributions, args = setup(args)
    param_names = [p["name"] for p in parameters]
    datasets = dataset_names(args)

    pk_k = (
        None
        if args.no_pk
        else np.geomspace(args.pk_kmin, args.pk_kmax, args.pk_nk)
    )

    # Every esub job rebuilds the hypercube and takes its own slice, so it must be
    # the same one: np.random.seed does not reach scipy.stats.qmc (SciPy >= 1.15),
    # which made each index of the Aug 2026 run an independent random design.
    lhd = qmc.LatinHypercube(
        d=len(parameters), rng=np.random.default_rng(42)
    ).random(n=args.n_gridpoints)
    sampled_values = emulator.sample_from_hypercube(lhd, prior_ranges, distributions)

    for index in indices:
        start = index * args.num_samps_per_index
        end = start + args.num_samps_per_index
        sub_samples = sampled_values[start:end]

        spec_lists = {s: [] for s in datasets}
        params_list = []
        axis_grids = {}
        bad = []

        pbar = tqdm(range(sub_samples.shape[0]), desc=f"Index {index}", unit="sample")
        for i in pbar:
            params = sub_samples[i]
            params_dict = {param_names[j]: params[j] for j in range(len(param_names))}
            cosmo, bcm = utils.split_cosmo_baryon(params_dict)
            try:
                out = utils.predict_cmb_cls(
                    cosmo,
                    bcm,
                    lmax=args.lmax,
                    ell_min=args.ell_min,
                    feedback=args.feedback,
                    q2=args.q2,
                    kmax=args.kmax,
                    verbose=False,
                    return_ckk=not args.no_clkk,
                    pk_k=pk_k,
                    pk_z_stride=args.pk_z_stride,
                )
                for axis in AXES:
                    if axis not in axis_grids and axis in out:
                        axis_grids[axis] = np.asarray(out[axis])
                for s in datasets:
                    arr = np.asarray(out[s])
                    spec_lists[s].append(
                        arr.astype(np.float32) if s in FLOAT32_DATASETS else arr
                    )
                params_list.append(params)
            except Exception as e:  # noqa: BLE001 — skip CAMB/BCemu failures
                LOGGER.warning("index %s sample %s failed: %s", index, i, e)
                bad.append(i)
            pbar.set_postfix({"ok": len(params_list), "bad": len(bad)})

        LOGGER.info("index %s: %s ok, %s bad", index, len(params_list), len(bad))

        save_kwargs = {s: np.asarray(spec_lists[s]) for s in datasets}
        # tuple() per row is required: handing np.array a list of 1-D arrays with a
        # structured dtype broadcasts each value into every field, giving an
        # (n_samples, n_params) array in which all 14 fields hold the same vector
        # (and a 196-wide, 14x-tiled feature matrix downstream).
        save_kwargs["params"] = np.array(
            [tuple(p) for p in params_list],
            dtype=[(name, float) for name in param_names],
        )
        save_kwargs.update(axis_grids)
        np.savez(os.path.join(args.scratch, f"cls_{index}.npz"), **save_kwargs)
        yield index


def merge(indices, args, *, buffer_rows: int = 1000) -> None:
    """Collect per-index ``cls_<idx>.npz`` files into one ``total_data.h5``."""
    _, _, _, args = setup(args)
    # Datasets are resolved from the first npz that loads, so runs made with
    # --no_pk/--no_clkk (or by an older version of this script) still merge.
    data_io.merge_npz(
        [os.path.join(args.scratch, f"cls_{idx}.npz") for idx in indices],
        os.path.join(args.scratch, "total_data.h5"),
        dataset_names(args),
        AXES,
        buffer_rows=buffer_rows,
    )
