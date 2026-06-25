"""Latin-hypercube sampling and training-data cleaning helpers.

These are lightweight copies of the corresponding utilities in
``multiprobe_framework.emulator`` (functions ``sample_from_hypercube``,
``remove_nan_rows``, ``remove_zero_rows``), trimmed of the tensorflow /
cosmic_toolbox dependencies so the package stays self-contained.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import norm

LOGGER = logging.getLogger(__name__)


def sample_from_hypercube(lhc, prior_ranges, distributions, cov_file=None, mu_file=None):
    """Map a unit Latin-hypercube design onto the requested priors.

    :param lhc: Latin Hypercube samples in the unit cube, shape (n_samples, n_params).
    :param prior_ranges: List of tuples, one per parameter: ``(min, max)`` for a
        ``flat`` prior or ``(mu, sigma)`` for a ``gaussian`` prior.
    :param distributions: List of distribution names (``"flat"`` or ``"gaussian"``).
    :param cov_file: Unused (kept for signature compatibility).
    :param mu_file: Unused (kept for signature compatibility).
    :return: Sampled values, shape (n_samples, n_params).
    """
    num_samples, num_params = lhc.shape
    sampled_values = np.zeros((num_samples, num_params))

    for i in range(num_params):
        dist = distributions[i]
        if dist == "flat":
            lower_bound, upper_bound = prior_ranges[i]
            parameter_range = upper_bound - lower_bound
            sampled_values[:, i] = (lhc[:, i] * parameter_range) + lower_bound

        elif dist == "gaussian":
            mu, sigma = prior_ranges[i]
            sampled_values[:, i] = norm.ppf(lhc[:, i], loc=mu, scale=sigma)

        else:
            LOGGER.error(
                "Invalid distribution type '%s'. Choose from 'flat' or 'gaussian'.", dist
            )

    return sampled_values


def remove_nan_rows(y_train, x_train):
    """Drop rows of ``y_train`` containing any NaN (and the matching ``x_train`` rows)."""
    mask = ~np.isnan(y_train).any(axis=1)
    n_removed = int(np.sum(~mask))
    print(f"Removed {n_removed} rows with NaN values from the training data.")
    return y_train[mask], x_train[mask]


def remove_zero_rows(y_train, x_train):
    """Drop rows of ``y_train`` that are entirely zero (and the matching ``x_train`` rows)."""
    mask = ~np.all(y_train == 0, axis=1)
    n_removed = int(np.sum(~mask))
    print(f"Removed {n_removed} all-zero rows from the training data.")
    return y_train[mask], x_train[mask]
