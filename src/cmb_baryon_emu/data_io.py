"""Merge per-index training-data npz files into one HDF5 file.

Shared by the esub ``merge`` entry points of ``scripts/generate_training_data.py``
and ``scripts/generate_lss_training_data.py``.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import h5py
import numpy as np
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)

TARGET_CHUNK_KB = 3000


def merge_npz(npz_paths, h5_path, datasets, axes, attrs=None, *, buffer_rows=1000):
    """Stack the per-sample datasets of ``npz_paths`` row-wise into ``h5_path``.

    Parameters
    ----------
    npz_paths : list of str
        Per-index npz files, merged in this order; unreadable files are skipped.
    h5_path : str
        Output HDF5 file (overwritten).
    datasets : list of str
        Candidate per-sample dataset names. Those present in the first non-empty
        npz are merged, plus ``params``; files whose shapes or dtypes disagree with
        it are skipped.
    axes : list of str
        Shared coordinate arrays, written once from the first file that has them.
    attrs : dict, optional
        ``{dataset_or_axis: {attr: value}}`` written as HDF5 attributes (e.g. units).

    Returns
    -------
    int
        Number of rows written.
    """
    from hdf5plugin import Zstd

    codec = Zstd(clevel=2)
    attrs = attrs or {}
    names: list[str] = []
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
        dset = h5.require_dataset(
            name,
            shape=(0, *rest),
            maxshape=(None, *rest),
            chunks=(chunk_rows, *rest),
            dtype=sample.dtype,
            **codec,
        )
        dset.attrs.update(attrs.get(name, {}))
        return dset

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

    total_rows = 0
    axes_written = set()

    with h5py.File(h5_path, "w", libver="latest") as h5:
        for npz_path in tqdm(npz_paths, desc="merge"):
            try:
                with np.load(npz_path) as z:
                    if not dsets:
                        if z["params"].shape[0] == 0:
                            # An all-failed index carries no per-row shapes; taking
                            # the layout from it would mismatch every later file.
                            LOGGER.warning("skip %s: no samples", npz_path)
                            continue
                        names = [n for n in datasets if n in z] + ["params"]
                        for name in names:
                            dsets[name] = _require_once(h5, name, z[name])
                    for axis in axes:
                        if axis not in axes_written and axis in z:
                            h5.create_dataset(axis, data=z[axis])
                            h5[axis].attrs.update(attrs.get(axis, {}))
                            axes_written.add(axis)

                    skip = False
                    for name in names:
                        if (
                            z[name].shape[1:] != true_shape[name]
                            or z[name].dtype != true_dtype[name]
                        ):
                            LOGGER.warning("skip %s: %s shape mismatch", npz_path, name)
                            skip = True
                            break
                    if skip:
                        continue

                    for name in names:
                        buffers[name].append(np.asarray(z[name]))
                        if sum(b.shape[0] for b in buffers[name]) >= buffer_rows:
                            _flush(name)
                    total_rows += z["params"].shape[0]
            except Exception as e:  # noqa: BLE001
                LOGGER.warning("could not load %s: %s", npz_path, e)

        for name in names:
            _flush(name)

    LOGGER.info("merge complete -> %s (%d rows)", h5_path, total_rows)
    print(f"merge complete -> {h5_path} ({total_rows} rows)")
    return total_rows
