# cmb_baryon_emu

A lightweight, self-contained package to (1) **predict** lensed & unlensed CMB
`C_ℓ` (TT, TE, EE) up to `ℓ = 10000` from cosmological + baryon parameters by
wrapping [BCemu](../BCemu)'s spectra module, (2) **generate** Latin-hypercube
training data, and (3) **train** neural-network emulators of those spectra. A
pure-JAX inference engine is bundled for fast, differentiable prediction after
training.

It has **no dependency on UPanda / multiprobe_framework** — the emulator engine
(`integrated_model.py`, `integrated_model_jax.py`, and the `module.py`/`config.py`
numpy↔jax switch) is copied in and self-contained.

## Install

```bash
# the BCemu fork with the spectra module (provides BaryonicCAMB / BCemuCAMB + CAMB)
pip install -e ../BCemu[spectra]

# this package (prediction + data-gen). Add extras as needed:
pip install -e .            # prediction + generate/merge
pip install -e .[train]     # + TensorFlow training engine
pip install -e .[jax]       # + JAX/Flax inference engine
```

## Predict

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

`total_data.h5` contains one dataset per spectrum
(`TT_lensed, TT_unlensed, EE_lensed, EE_unlensed, TE_lensed, TE_unlensed`), plus
`params` (structured array) and `ell`.

## Use a trained model

```python
from cmb_baryon_emu import JaxNpzModel

m = JaxNpzModel.from_package("tt_lensed")
cl = m.predict_matrix(params_mat)  # params_mat has the 196-feature training layout
```

The package ships lightweight JAX/NPZ model exports in
`src/cmb_baryon_emu/lightweight_models/`. To regenerate them from legacy
TensorFlow/Keras saves in `saved_models/`, run:

```bash
python scripts/convert_saved_models_to_npz.py --canonical-spectra
```

## Validate accuracy

[`notebooks/validate_emulator_accuracy.ipynb`](notebooks/validate_emulator_accuracy.ipynb)
compares emulator predictions against the lightweight held-out fixture in
`data/validation_all_spectra_500.h5` using the packaged JAX/NPZ model. The fixture
contains 500 examples for all six CMB spectra, compressed to stay suitable for a
normal GitHub repository.

Full training data and legacy TensorFlow/Keras artifacts are intentionally not
committed: `total_data.h5` is a cluster-scale data product, and `saved_models/`
is replaced by the lightweight package NPZ exports.

## Layout

```
src/cmb_baryon_emu/
  utils.py                 # predict_cmb_cls, load_config, cosmo_to_bcemu
  emulator.py              # LHC sampling + row cleaning helpers
  integrated_model.py      # TF/Keras emulator engine (copied, standalone)
  integrated_model_jax.py  # JAX/Flax engine (same saved format)
  jax_emu_model.py         # lightweight packaged JAX/NPZ inference
  lightweight_models/      # committed compact model weights
  module.py / config.py    # numpy<->jax backend switch (copied, standalone)
configs/   scripts/   submission_scripts/   notebooks/
```
