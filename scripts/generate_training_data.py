"""Generate CMB + baryon training data for the cmb_baryon_emu emulator.

Samples cosmological + BCemu2025 baryon parameters on a Latin hypercube and
computes lensed & unlensed TT/TE/EE C_ell (via cmb_baryon_emu.utils ->
BCemu.BCemuCAMB) for each point. esub entry points: ``main`` (per-index
generation) and ``merge`` (collect per-index npz into one HDF5).

Modelled on multiprobe_framework/scripts/generate_emulator_training_data.py.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
from collections import defaultdict

import h5py
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

from cmb_baryon_emu import emulator, utils  # noqa: E402

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

    np.random.seed(42)
    lhd = qmc.LatinHypercube(d=len(parameters)).random(n=args.n_gridpoints)
    sampled_values = emulator.sample_from_hypercube(lhd, prior_ranges, distributions)

    for index in indices:
        start = index * args.num_samps_per_index
        end = start + args.num_samps_per_index
        sub_samples = sampled_values[start:end]

        spec_lists = {s: [] for s in SPECTRA}
        params_list = []
        ell_grid = None
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
                )
                if ell_grid is None:
                    ell_grid = out["ell"]
                for s in SPECTRA:
                    spec_lists[s].append(out[s])
                params_list.append(params)
            except Exception as e:  # noqa: BLE001 — skip CAMB/BCemu failures
                LOGGER.warning("index %s sample %s failed: %s", index, i, e)
                bad.append(i)
            pbar.set_postfix({"ok": len(params_list), "bad": len(bad)})

        LOGGER.info("index %s: %s ok, %s bad", index, len(params_list), len(bad))

        save_kwargs = {s: np.asarray(spec_lists[s]) for s in SPECTRA}
        save_kwargs["params"] = np.array(
            params_list, dtype=[(name, float) for name in param_names]
        )
        if ell_grid is not None:
            save_kwargs["ell"] = np.asarray(ell_grid)
        np.savez(os.path.join(args.scratch, f"cls_{index}.npz"), **save_kwargs)
        yield index


def merge(indices, args, *, buffer_rows: int = 1000) -> None:
    """Collect per-index ``cls_<idx>.npz`` files into one ``total_data.h5``."""
    _, _, _, args = setup(args)

    from hdf5plugin import Zstd

    CODEC = Zstd(clevel=2)
    TARGET_CHUNK_KB = 3000
    DATASETS = SPECTRA + ["params"]

    true_shape: dict[str, tuple] = {}
    true_dtype: dict[str, np.dtype] = {}
    buffers: dict[str, list] = defaultdict(list)
    offsets: dict[str, int] = defaultdict(int)
    dsets: dict[str, h5py.Dataset] = {}

    def _require_once(h5, name, sample):
        rows, *rest = sample.shape
        true_shape[name] = tuple(rest)
        true_dtype[name] = sample.dtype
        row_bytes = sample.dtype.itemsize * (int(np.prod(rest)) or 1)
        chunk_rows = max(1, (TARGET_CHUNK_KB * 1024) // row_bytes)
        return h5.require_dataset(
            name,
            shape=(0, *rest),
            maxshape=(None, *rest),
            chunks=(chunk_rows, *rest),
            dtype=sample.dtype,
            **CODEC,
        )

    def _flush(name):
        if not buffers[name]:
            return
        data = np.concatenate(buffers[name], axis=0)
        dset = dsets[name]
        new_size = offsets[name] + data.shape[0]
        dset.resize((new_size, *true_shape[name]))
        dset[offsets[name] : new_size, ...] = data
        offsets[name] += data.shape[0]
        buffers[name].clear()

    h5_path = os.path.join(args.scratch, "total_data.h5")
    total_rows = 0
    ell_written = False

    with h5py.File(h5_path, "w", libver="latest") as h5:
        for idx in tqdm(indices, desc="merge"):
            npz_path = os.path.join(args.scratch, f"cls_{idx}.npz")
            try:
                with np.load(npz_path) as z:
                    if not dsets:
                        for name in DATASETS:
                            dsets[name] = _require_once(h5, name, z[name])
                    if not ell_written and "ell" in z:
                        h5.create_dataset("ell", data=z["ell"])
                        ell_written = True

                    skip = False
                    for name in DATASETS:
                        if (
                            z[name].shape[1:] != true_shape[name]
                            or z[name].dtype != true_dtype[name]
                        ):
                            LOGGER.warning("skip %s: %s shape mismatch", npz_path, name)
                            skip = True
                            break
                    if skip:
                        continue

                    for name in DATASETS:
                        buffers[name].append(np.asarray(z[name]))
                        if sum(b.shape[0] for b in buffers[name]) >= buffer_rows:
                            _flush(name)
                    total_rows += z["params"].shape[0]
            except Exception as e:  # noqa: BLE001
                LOGGER.warning("could not load %s: %s", npz_path, e)

        for name in DATASETS:
            _flush(name)

    LOGGER.info("merge complete -> %s (%d rows)", h5_path, total_rows)
    print(f"merge complete -> {h5_path} ({total_rows} rows)")
