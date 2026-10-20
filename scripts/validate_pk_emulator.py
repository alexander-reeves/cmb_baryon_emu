"""Accuracy test for a trained cmb_baryon_emu emulator on held-out samples.

Written for the P(k) emulators (pk_lin / pk_nl / pk_dmb), whose targets live on a
(n_z, n_k) grid, but it also handles the 1-D spectra (TT/EE/TE/clkk) so one tool
covers every model.

The rows used are strictly after the ones seen in training: the P(k) models write
an ``*_axes.npz`` sidecar recording ``ntrain``, and this script starts there by
default. For a 1-D spectrum pass ``--start_row`` explicitly.

Example
-------
    python scripts/validate_pk_emulator.py \\
        --training_data_file /cluster/scratch/areeves/cmb_baryon_td/total_data.h5 \\
        --model_dir saved_models --model_name pk_nl \\
        --n_valid 5000 --plot pk_nl_accuracy.png
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 — registers the Zstd decompressor
import numpy as np

LOGGER = logging.getLogger(__name__)

# Reported everywhere, so a run is comparable against the emulator target.
PERCENTILES = [50.0, 68.0, 95.0, 99.0, 99.9]

# k bands [Mpc^-1] the P(k) report is split over: large scales, the BAO/transition
# region, the one-halo regime, and the deep nonlinear tail that motivated kmax=50.
# The bands are in the grid's own unit (Mpc^-1 for the Aug 2026 run, h/Mpc for
# generate_lss_training_data.py).
K_BANDS = [(1e-5, 1e-2), (1e-2, 0.3), (0.3, 1.0), (1.0, 10.0), (10.0, 50.0 + 1e-9)]


def setup(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--training_data_file", required=True)
    p.add_argument("--model_dir", required=True,
                   help="directory holding <model_name>_model.h5 / .pkl / _axes.npz")
    p.add_argument("--model_name", required=True)
    p.add_argument("--spectrum", default=None,
                   help="HDF5 dataset to compare against (default: from the axes "
                        "sidecar, else the model name)")
    p.add_argument("--n_valid", type=int, default=5000,
                   help="number of held-out samples to test (default 5000)")
    p.add_argument("--start_row", type=int, default=None,
                   help="first row to test (default: ntrain from the axes sidecar)")
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--tol_p95", type=float, default=0.01,
                   help="p95 |rel. error| above which the run is reported as FAIL")
    p.add_argument("--plot", default=None, help="write a PNG summary figure here")
    p.add_argument("--json", default=None, help="write the full report as JSON here")
    p.add_argument("--verbosity", default="info")
    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.verbosity.upper(), logging.INFO),
                        format="%(message)s")
    return args


def load_axes(model_dir, model_name):
    """Read the P(k) axes sidecar written by train_emulator.py, if present."""
    path = Path(model_dir) / f"{model_name}_axes.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return dict(
            spectrum=str(z["spectrum"]),
            shape=tuple(int(v) for v in z["shape"]),
            pk_k=z["pk_k"],
            pk_z=z["pk_z"],
            parameter_names=[str(s) for s in z["parameter_names"]],
            ntrain=int(z["ntrain"]) if "ntrain" in z else None,
            k_unit=str(z["k_unit"]) if "k_unit" in z else "Mpc^-1",
            ratio_to=str(z["ratio_to"]) if "ratio_to" in z else "",
        )


def predict_in_batches(model, x_struct, parameter_names, batch_size):
    """Run IntegratedModel.predict over the held-out rows, batched to bound memory."""
    out = []
    n = x_struct.shape[0]
    for start in range(0, n, batch_size):
        chunk = x_struct[start:start + batch_size]
        data_dict = {name: np.asarray(chunk[name], dtype=float) for name in parameter_names}
        out.append(np.asarray(model.predict(data_dict)))
    return np.concatenate(out, axis=0)


def summarize(abs_rel):
    """Percentile summary of |pred/truth - 1| over every element given."""
    q = np.percentile(abs_rel, PERCENTILES)
    return {
        **{f"p{p:g}": float(v) for p, v in zip(PERCENTILES, q)},
        "max": float(abs_rel.max()),
        "frac_gt_0.1pct": float((abs_rel > 1e-3).mean()),
        "frac_gt_1pct": float((abs_rel > 1e-2).mean()),
    }


def _fmt(stats):
    return (f"{stats['p50']:9.2e} {stats['p68']:9.2e} {stats['p95']:9.2e} "
            f"{stats['p99']:9.2e} {stats['max']:9.2e}")


def main(argv=None):
    args = setup(argv)

    axes = load_axes(args.model_dir, args.model_name)
    spectrum = args.spectrum or (axes["spectrum"] if axes else args.model_name)
    start_row = args.start_row
    if start_row is None:
        if axes is None or axes["ntrain"] is None:
            raise SystemExit(
                "no axes sidecar with ntrain: pass --start_row so the test rows are "
                "provably outside the training set"
            )
        start_row = axes["ntrain"]

    # Import here so --help works without a TensorFlow load.
    from cmb_baryon_emu import integrated_model

    model = integrated_model.IntegratedModel(None, None, None, None)
    model.restore(str(Path(args.model_dir) / args.model_name))
    parameter_names = list(model.parameter_names)
    LOGGER.info("model %s: %d inputs %s", args.model_name, len(parameter_names),
                parameter_names)

    with h5py.File(args.training_data_file, "r") as f:
        n_total = f["params"].shape[0]
        stop = min(start_row + args.n_valid, n_total)
        if start_row >= n_total:
            raise SystemExit(f"start_row {start_row} >= {n_total} rows in file")
        n_valid = stop - start_row
        LOGGER.info("testing rows %d..%d (%d samples) of %d, spectrum=%s",
                    start_row, stop, n_valid, n_total, spectrum)
        x_struct = f["params"][start_row:stop]
        truth = f[spectrum][start_row:stop].astype(np.float64)
        if axes is not None and axes["ratio_to"]:
            # the model predicts spectrum / ratio_to (e.g. the nonlinear boost)
            truth /= f[axes["ratio_to"]][start_row:stop].astype(np.float64)
            spectrum = f"{spectrum}/{axes['ratio_to']}"

    pred = predict_in_batches(model, x_struct, parameter_names, args.batch_size)

    if axes is not None:
        pred = pred.reshape(-1, *axes["shape"])
    if pred.shape != truth.shape:
        raise SystemExit(f"prediction shape {pred.shape} != truth {truth.shape}")

    # Relative error is the right metric here: P(k) spans ~7 decades, so an
    # absolute residual is meaningless across the grid.
    rel = pred / truth - 1.0
    abs_rel = np.abs(rel)

    report = {
        "model": args.model_name,
        "spectrum": spectrum,
        "rows": [int(start_row), int(stop)],
        "n_valid": int(n_valid),
        "parameter_names": parameter_names,
        "overall": summarize(abs_rel),
    }

    print(f"\n=== {args.model_name} ({spectrum}) — {n_valid} held-out samples, "
          f"rows {start_row}..{stop} ===")
    print(f"{'':22s} {'p50':>9s} {'p68':>9s} {'p95':>9s} {'p99':>9s} {'max':>9s}")
    print(f"{'overall':22s} {_fmt(report['overall'])}")

    if axes is not None:
        z_grid, k_grid = axes["pk_z"], axes["pk_k"]
        # per-redshift: collapses samples and k
        per_z = [summarize(abs_rel[:, i, :]) for i in range(len(z_grid))]
        report["per_z"] = {f"{z:.3f}": s for z, s in zip(z_grid, per_z)}
        print("\nby redshift:")
        for z, s in zip(z_grid, per_z):
            print(f"  z = {z:<18.3f} {_fmt(s)}")

        # per-k band: collapses samples and z
        print(f"\nby k band [{axes['k_unit']}]:")
        report["per_k_band"] = {}
        for lo, hi in K_BANDS:
            m = (k_grid >= lo) & (k_grid < hi)
            if not m.any():
                continue
            s = summarize(abs_rel[:, :, m])
            report["per_k_band"][f"{lo:g}-{hi:g}"] = s
            print(f"  {lo:8.4g} - {hi:<8.4g}      {_fmt(s)}")

        # worst grid cell, to point at the failure mode rather than just its size
        p95_grid = np.percentile(abs_rel, 95, axis=0)          # (n_z, n_k)
        iz, ik = np.unravel_index(int(np.argmax(p95_grid)), p95_grid.shape)
        report["worst_cell"] = {"z": float(z_grid[iz]), "k": float(k_grid[ik]),
                                "p95": float(p95_grid[iz, ik])}
        print(f"\nworst (z,k) cell by p95: z={z_grid[iz]:.3f}, "
              f"k={k_grid[ik]:.4g} {axes['k_unit']} -> p95={p95_grid[iz, ik]:.2e}")
    else:
        p95_grid = np.percentile(abs_rel, 95, axis=0)
        report["worst_bin"] = {"index": int(np.argmax(p95_grid)),
                               "p95": float(p95_grid.max())}
        print(f"\nworst output bin by p95: index {int(np.argmax(p95_grid))} "
              f"-> {p95_grid.max():.2e}")

    ok = report["overall"]["p95"] <= args.tol_p95
    report["tol_p95"] = args.tol_p95
    report["pass"] = bool(ok)
    print(f"\n{'PASS' if ok else 'FAIL'}: overall p95 = "
          f"{report['overall']['p95']:.3e} (tolerance {args.tol_p95:.3e})")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        LOGGER.info("wrote %s", args.json)
    if args.plot:
        make_plot(abs_rel, axes, args.model_name, spectrum, args.plot)
        LOGGER.info("wrote %s", args.plot)

    return 0 if ok else 1


def make_plot(abs_rel, axes, model_name, spectrum, path):
    """Percentile bands vs k, and the p95 error over the full (z, k) grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if axes is None:  # 1-D spectrum: percentile bands vs output index only
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        x = np.arange(abs_rel.shape[1])
        # ordered percentiles -> single-hue sequential ramp, light to dark
        shades = plt.cm.Blues(np.linspace(0.45, 0.95, len(PERCENTILES)))
        for p, c in zip(PERCENTILES, shades):
            ax.plot(x, np.percentile(abs_rel, p, axis=0), color=c, lw=2, label=f"p{p:g}")
        ax.set_yscale("log")
        ax.set_xlabel("output index")
        ax.set_ylabel(r"$|\hat{y}/y - 1|$")
        ax.set_title(f"{model_name} ({spectrum}) — held-out accuracy")
        ax.legend(frameon=False, ncol=len(PERCENTILES), fontsize=8)
        ax.grid(alpha=0.25, lw=0.6)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return

    k, z = axes["pk_k"], axes["pk_z"]
    k_label = r"$k\ [h\,\mathrm{Mpc}^{-1}]$" if axes["k_unit"] == "h/Mpc" else r"$k\ [\mathrm{Mpc}^{-1}]$"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)

    # Left: percentile bands vs k (collapsed over samples and z). Ordered
    # magnitudes -> one hue, light to dark; no second y-axis.
    shades = plt.cm.Blues(np.linspace(0.4, 0.95, len(PERCENTILES)))
    for p, c in zip(PERCENTILES, shades):
        ax1.plot(k, np.percentile(abs_rel, p, axis=(0, 1)), color=c, lw=2, label=f"p{p:g}")
    ax1.axhline(1e-2, color="0.35", lw=1, ls="--")
    ax1.text(k[0], 1.15e-2, "1%", color="0.35", fontsize=8, va="bottom")
    ax1.axhline(1e-3, color="0.55", lw=1, ls=":")
    ax1.text(k[0], 1.15e-3, "0.1%", color="0.55", fontsize=8, va="bottom")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(k_label)
    ax1.set_ylabel(r"$|P_{\rm emu}/P_{\rm CAMB} - 1|$")
    ax1.set_title(f"{model_name} — error vs scale")
    ax1.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
    ax1.grid(alpha=0.25, lw=0.6)

    # Right: p95 over the (z, k) grid. Magnitude -> sequential colormap.
    p95 = np.percentile(abs_rel, 95, axis=0)
    mesh = ax2.pcolormesh(k, z, p95, shading="nearest", cmap="magma_r",
                          norm=matplotlib.colors.LogNorm(
                              vmin=max(p95.min(), 1e-6), vmax=p95.max()))
    ax2.set_xscale("log")
    ax2.set_xlabel(k_label)
    ax2.set_ylabel(r"$z$")
    ax2.set_title("p95 relative error over the grid")
    cb = fig.colorbar(mesh, ax=ax2)
    cb.set_label("p95 $|$rel. error$|$")

    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
