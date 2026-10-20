#!/bin/bash
# Copy the merged LSS training set off scratch, after checking it is complete.
#
# The merge step skips per-index files that are missing or unreadable, so a merged
# total_data.h5 can exist even when generation tasks failed. Exit non-zero (so that
# trainings submitted with --dependency=afterok:<this job> never start) unless it has
# at least MIN_ROWS samples. 500k LHC points with ~3 % Omega_m > 1 skipped -> ~485k.
#
#   sbatch --dependency=afterok:<merge job id> submission_scripts/stage_lss_data.sh
#
#SBATCH --job-name=stage_lss_data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=04:00:00
#SBATCH --output=esub_logs/stage_lss_data_%j.out

set -euo pipefail

SRC=${SRC:-/cluster/scratch/areeves/cmb_baryon_lss_td/total_data.h5}
DST=${DST:-/cluster/work/refregier/alexree/local_packages/cmb_baryon_emu/data/lss_total_data.h5}
MIN_ROWS=${MIN_ROWS:-470000}

source /cluster/work/refregier/alexree/local_packages/cmb_baryon_emu/submission_scripts/setup_env.sh

python3 - "$SRC" "$MIN_ROWS" <<'EOF'
import sys
import h5py
import hdf5plugin  # noqa: F401 — registers the Zstd decompressor
import numpy as np

path, min_rows = sys.argv[1], int(sys.argv[2])
with h5py.File(path, "r") as f:
    n = {k: f[k].shape[0] for k in ("params", "pk_lin", "pk_nl", "clpp", "clpp_dmo")}
    print(f"{path}: rows {n}")
    if len(set(n.values())) != 1:
        sys.exit("row counts differ between datasets")
    rows = n["params"]
    if rows < min_rows:
        sys.exit(f"only {rows} rows (< {min_rows}): generation tasks failed")
    # spot-check the last block for NaN/inf, where a broken task would show up
    tail = slice(max(0, rows - 2000), rows)
    for k in ("pk_lin", "pk_nl", "clpp"):
        x = f[k][tail]
        if not np.all(np.isfinite(x)) or np.any(x <= 0):
            sys.exit(f"{k}: non-finite or non-positive values in the last 2000 rows")
print("check passed")
EOF

mkdir -p "$(dirname "$DST")"
rsync -a "$SRC" "$DST"
echo "copied to $DST ($(du -h "$DST" | cut -f1))"
