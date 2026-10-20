#!/bin/bash
# Environment for cmb_baryon_emu generation/training jobs.
#
# Source this before running epipe, and pass it to esub via --source_file so the
# SLURM jobs get the same interpreter:
#   source submission_scripts/setup_env.sh
#   python3 -m esub.epipe submission_scripts/submit_generate.yaml
#
# Why a dedicated venv: BCemu's spectra module defaults to lensing_method='camb'
# (commit b4f0379, 2026-07-17), which needs camb>=1.6.6 for
# camb.nonlinear.ExternalNonLinearRatio. The mcmc_jax_new venv cannot provide it
# -- its `camb` is an editable MGCamb 1.5.2 that shadows any pip install -- so
# BCemuCAMB raises ImportError there.

source /cluster/project/refregier/areeves/cmb_baryon_venv/bin/activate

# GPU training: tensorflow[and-cuda] 2.21 ships the CUDA libraries as nvidia-* wheels,
# but libtensorflow_framework's RUNPATH misses nvidia/cusolver/lib, so TF cannot
# dlopen libcusolver.so.11 and silently trains on CPU ("Skipping registering GPU
# devices"). Put every CUDA wheel's lib directory on the search path.
for _d in /cluster/project/refregier/areeves/cmb_baryon_venv/lib/python3.11/site-packages/nvidia/*/lib; do
    LD_LIBRARY_PATH="$_d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
export LD_LIBRARY_PATH
unset _d
