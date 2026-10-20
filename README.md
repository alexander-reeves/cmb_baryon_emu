# cmb_baryon_emu

A lightweight, self-contained package to (1) **predict** lensed & unlensed CMB
`C_ℓ` (TT, TE, EE) up to `ℓ = 10000` from cosmological + baryon parameters by
wrapping [BCemu](../BCemu)'s spectra module, (2) **generate** Latin-hypercube
training data, and (3) **train** neural-network emulators of those spectra. A
pure-JAX inference engine is bundled for fast, differentiable prediction after
training.

## Install

On Euler there is a ready-made venv with the whole stack (BCemu + camb 2.0.1 +
this package + esub); the submission YAMLs point SLURM jobs at it via
`--source_file=submission_scripts/setup_env.sh`:

```bash
source submission_scripts/setup_env.sh   # /cluster/project/refregier/areeves/cmb_baryon_venv
```

Do **not** use the `mcmc_jax_new` venv here: its `camb` is an editable MGCamb
1.5.2 that shadows any pip install, and BCemu's default `lensing_method='camb'`
needs `camb>=1.6.6` (`camb.nonlinear.ExternalNonLinearRatio`), so `BCemuCAMB`
raises `ImportError` there.

To build the environment from scratch:

```bash
# the BCemu fork with the spectra module (provides BaryonicCAMB / BCemuCAMB + CAMB)
pip install -e ../BCemu[spectra]

# this package (prediction + data-gen). Add extras as needed:
pip install -e .            # prediction + generate/merge
pip install -e .[train]     # + TensorFlow training engine
pip install -e .[jax]       # + JAX/Flax inference engine
```

## Make theory prediction (no emulator)

```python
from cmb_baryon_emu import utils

cosmo = dict(h=0.6736, omega_b=0.02237, omega_cdm=0.1200,
             ln10As=3.044, n_s=0.9649, tau=0.0544)          # 6 LCDM params
bcm   = dict(log10Mc=13.1, Theta_co=0.3, mu=1.0, delta=6.0,  # 8 BCemu2025 params
             eta=0.10, deta=0.22, Nstar=0.028, fb=0.0486/0.306)

out = utils.predict_cmb_cls(cosmo, bcm, lmax=10000)
out["ell"]            # ell = 2 .. 10000
out["TT_lensed"], out["TT_unlensed"]     # muK^2, raw C_ell
out["TE_lensed"], out["EE_lensed"], ...  # TE_*, EE_* likewise
```

The same call can also hand back the CMB lensing spectrum and the matter power
spectra. Both are by-products of the CAMB solves `BCemuCAMB` runs anyway, so they
are essentially free (spline evaluation only, no extra CAMB call):

```python
out = utils.predict_cmb_cls(cosmo, bcm, lmax=10000, return_ckk=True, pk_k=True)
out["clkk"]                       # C_L^kappakappa on the same ell grid
out["pk_k"], out["pk_z"]          # k [Mpc^-1] (1e-4 .. 50, 300 pts), z (0 .. 5, 21 pts)
out["pk_lin"], out["pk_nl"]       # (n_z, n_k) Mpc^3, dark-matter-only (mead2020 NL)
out["pk_dmb"]                     # = pk_nl * S(k,z), the baryon-suppressed one
```

`pk_lin`/`pk_nl` depend on cosmology only; the 8 baryon parameters enter `pk_dmb`
alone. BCemu2025's suppression is defined for z <= 3.04 and 0.034-12.5 h/Mpc, and
falls back to S = 1 outside that, where `pk_dmb == pk_nl`. `pk_z` defaults to
BCemu's own nonlinear redshift nodes (so z is never interpolated); `pk_k` is a
fixed grid because CAMB's native k sampling shifts with cosmology.

## Train an emulator

The varied parameters and their priors live in
[`configs/emulator_config_cmb_baryon.yaml`](configs/emulator_config_cmb_baryon.yaml)
(6 cosmology + 8 baryon). The workflow mirrors `multiprobe_framework` and runs via
`esub`/`epipe`:

```bash
# 1. generate + merge training data -> $scratch/total_data.h5
python3 -m esub.epipe submission_scripts/submit_generate.yaml

# 2. train the 6 per-spectrum emulators (TT/EE log, TE PCA) on GPU
python3 -m esub.epipe submission_scripts/submit_train.yaml
```

`total_data.h5` contains one `(n_samples, n_ell)` dataset per spectrum
(`TT_lensed, TT_unlensed, EE_lensed, EE_unlensed, TE_lensed, TE_unlensed`) plus
`clkk` on the same `ell` axis, three `(n_samples, n_z, n_k)` matter power spectra
in Mpc^3 (`pk_lin, pk_nl, pk_dmb`, float32), the shared axes `ell`, `pk_k`
(Mpc^-1) and `pk_z`, and `params` (structured array). Pass `--no_pk` / `--no_clkk`
to `generate_training_data.py` to drop the extra datasets; `merge` takes whatever
the per-index npz files actually contain.

### Matter power spectrum emulators

`pk_lin`, `pk_nl` and `pk_dmb` train from the same HDF5. Their targets are
`(21, 300)` per sample, which `train_emulator.py` flattens to a 6300-wide output
and records in a `<model_name>_axes.npz` sidecar (`pk_k`, `pk_z`, `shape`,
`parameter_names`, `ntrain`) so a prediction can be reshaped back:

```python
import numpy as np
from cmb_baryon_emu import integrated_model

axes = np.load("saved_models/pk_nl_axes.npz")
m = integrated_model.IntegratedModel(None, None, None, None)
m.restore("saved_models/pk_nl")
pk = m.predict({n: vals for n, vals in ...}).reshape(-1, *axes["shape"])  # (n, n_z, n_k)
```

`--input_params` defaults to `auto`: `pk_lin`/`pk_nl` are dark-matter-only, so they
are trained on the 6 cosmological parameters alone (the 8 baryon parameters enter
only through `pk_dmb`). `m.parameter_names` always states what a given model wants.

Accuracy on held-out samples:

```bash
python scripts/validate_pk_emulator.py \
    --training_data_file $scratch/total_data.h5 \
    --model_dir saved_models --model_name pk_nl \
    --n_valid 5000 --plot pk_nl_accuracy.png --json pk_nl_accuracy.json
```

It starts at the `ntrain` recorded in the sidecar, so the test rows are provably
unseen, and reports |P_emu/P_CAMB − 1| percentiles overall, per redshift and per
k band, plus the worst (z, k) cell. Exit status is non-zero when the overall p95
exceeds `--tol_p95` (default 1%). It also handles the 1-D spectra (TT/EE/TE/clkk)
if given `--start_row` explicitly.

## P(k) + CMB lensing run

`scripts/generate_lss_training_data.py` builds a second training set for the matter
power spectra and C_L^phiphi, one CAMB run per sample via
`utils.predict_lss_spectra`:

| dataset | content | depends on |
|---|---|---|
| `pk_lin`, `pk_nl` | linear / HMcode-2020 P(k, z), dark-matter-only, (Mpc/h)^3, shape (31, 329) | cosmology only |
| `clpp` | raw C_L^phiphi, L = 2..3000, with BCemu2025 feedback | cosmology + baryons |
| `clpp_dmo` | the same without feedback | cosmology only |

Axes: `pk_k` (1e-5..50 h/Mpc, 329 points, 100/decade through the BAO), `pk_z`
(0..3, step 0.1), `ell`. The baryons enter C_L^phiphi as a Limber correction from
P_NL (S - 1); against a second CAMB run fed P_NL S through `ExternalNonLinearRatio`
this agrees to <= 4.4e-4 of C_L for L <= 3000, at about a third of the cost of
`BCemuCAMB`. Priors (`configs/emulator_config_lss_baryon.yaml`) follow the ACT DR6
lensing analysis (Madhavacheril et al. 2024, Table 1), with its very wide flat
ranges cut to the region its lensing-only chains reach.

```bash
python3 -m esub.epipe submission_scripts/submit_generate_lss.yaml
python3 -m esub.epipe submission_scripts/submit_train_lss.yaml   # pk_lin, pk_nl_boost, clpp
python scripts/check_lss_onerun.py                              # one-run vs two-run reference
```

`pk_nl_boost` predicts P_NL/P_lin (its sidecar records `ratio_to = pk_lin`), so
P_NL = pk_lin prediction x pk_nl_boost prediction.

The merge skips per-index files that are missing, so check the row count before
training; `submission_scripts/stage_lss_data.sh` does that (at least 470k rows, finite
spectra) and then copies `total_data.h5` off scratch. Submit it with
`--dependency=afterok:<merge job>`, and the trainings with `afterok` on it.

The packaged `pk_lin`, `pk_nl_boost` and `clpp` were trained in Oct 2026 on 440k of
483,751 samples. On the 43,751 held-out samples, |emu/CAMB - 1| is:

| model | p50 | p95 | p99 | max |
|---|---|---|---|---|
| `pk_lin` | 8.5e-5 | 4.6e-4 | 9.7e-4 | 1.0e-2 |
| `pk_nl_boost` (the ratio) | 2.4e-5 | 2.4e-4 | 4.5e-4 | 6.3e-3 |
| `clpp` | 3.0e-4 | 1.7e-3 | 2.8e-3 | 1.2e-2 |

P_NL from the product does as well as `pk_lin` (p95 4.6e-4 on 5000 of those samples).
The P(k) errors peak around the BAO, at k ~ 0.3-0.5 h/Mpc; the `clpp` errors grow
with L and are largest near L = 3000.
[`notebooks/load_lss_emulators.ipynb`](notebooks/load_lss_emulators.ipynb) shows how
to load and call the three models and compares them against one CAMB run.

## Use a trained model

```python
from cmb_baryon_emu import JaxNpzModel

m = JaxNpzModel.from_package("tt_lensed")
cl = m.predict_matrix(params_mat)  # see the feature-layout note below
```

**Feature layout.** Models trained on data generated *before* Aug 2026 expect a
**196**-column input: `generate_training_data.py` built its structured `params`
array from a list of 1-D arrays, which broadcast every value into all 14 fields,
so `params[<any field>]` returned the whole 14-vector and the trainer's
`column_stack` tiled the 14 real parameters 14 times. The models shipped in
`lightweight_models/` are from that era and still need the 196-wide matrix.
Data generated after the fix stores `params` with shape `(n_samples,)` and correct
per-field values, so models trained on it take the **14** parameters directly, in
the `configs/emulator_config_cmb_baryon.yaml` order. Check `m.parameter_names`
if unsure which vintage a model is.

The P(k) and lensing models take their parameters by name:

```python
pk_lin = JaxNpzModel.from_package("pk_lin")
boost = JaxNpzModel.from_package("pk_nl_boost")
clpp = JaxNpzModel.from_package("clpp")

plin = pk_lin.predict_pk(cosmo)          # (n, 31, 329) on pk_lin.pk_z x pk_lin.pk_k [h/Mpc]
pnl = plin * boost.predict_pk(cosmo)
cl = clpp.predict({**cosmo, **bcm})      # (n, 2999), L = 2..3000
```

The package ships lightweight JAX/NPZ model exports in
`src/cmb_baryon_emu/lightweight_models/`. To regenerate them from legacy
TensorFlow/Keras saves in `saved_models/`, run:

```bash
python scripts/convert_saved_models_to_npz.py --canonical-spectra
python scripts/convert_saved_models_to_npz.py --input-dir saved_models_lss \
    --model pk_lin --model pk_nl_boost --model clpp
```

A P(k) export also carries the k and z grids, the (n_z, n_k) shape and the ratio
denominator from the `_axes.npz` sidecar (`m.pk_k`, `m.pk_z`, `m.pk_axes`).

## Validate accuracy

[`notebooks/validate_emulator_accuracy.ipynb`](notebooks/validate_emulator_accuracy.ipynb)
compares emulator predictions against the lightweight held-out fixture in
`data/validation_all_spectra_500.h5` using the packaged JAX/NPZ model. The fixture
contains 500 examples for all six CMB spectra, compressed to stay suitable for a
normal GitHub repository. Full training data available on request. 


## Questions 

For questions or feedback, please contact Alexander Reeves: areeves@phys.ethz.ch