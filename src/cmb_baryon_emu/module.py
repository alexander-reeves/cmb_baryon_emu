"""Module-level imports for NumPy/JAX switching.

This module provides a unified interface for numpy/jax operations,
allowing the package to switch between NumPy and JAX backends seamlessly.
When JAX is enabled, all numpy operations are replaced with jax.numpy
equivalents for JIT compilation and automatic differentiation support.

The module exports numpy/jax functions and classes so they can be imported
as ``from cmb_baryon_emu.module import array, zeros, ...`` instead of importing
directly from numpy or jax.numpy.

Note:
    This module is intentionally imported via ``from cmb_baryon_emu.module import *``
    throughout UPanda. This is the mechanism that enables the JAX/NumPy
    backend switch: depending on the value of ``config.is_jax_enabled``,
    either jax.numpy or numpy symbols are exported.

    ``__all__`` is defined explicitly below to make this wildcard export
    deliberate and auditable. Future migration path: ``array-api-compat``
    (used by SciPy/scikit-learn) can eliminate the need for this module
    by auto-detecting the backend from input array types.
"""

import logging

import cmb_baryon_emu.config as config

logger = logging.getLogger(__name__)

is_jax = config.get_jax_enabled()

if is_jax:
    logger.debug("cmb_baryon_emu.module: Using JAX backend (jax.numpy)")
    import jax.numpy as numpy

    from jax.numpy import (
        isnan,
        isinf,
        savez,
        load,
        array,
        ndarray,
        conj,
        ones,
        tan,
        log,
        log10,
        logspace,
        swapaxes,
        empty,
        linspace,
        arange,
        delete,
        where,
        pi,
        cos,
        sin,
        exp,
        sqrt,
        concatenate,
        linalg,
        eye,
        einsum,
        einsum_path,
        zeros,
        sum,
        pad,
        diag,
        block,
        array_equal,
        meshgrid,
        geomspace,
        moveaxis,
        ones_like,
        empty_like,
        real,
        zeros_like,
        float32,
        float64,
        int32,
        dot,
        multiply,
        add,
        subtract,
        unique,
        hstack,
        isin,
        newaxis,
        ix_,
        transpose,
        interp,
        rollaxis,
        logical_or,
        inf,
        tanh,
        column_stack,
        power,
        asarray,
        sign,
        matmul,
        logical_and,
        stack,
        all,
        any,
        mean,
        ndim,
        ceil,
        argwhere,
        cov,
        nansum,
        nanmax,
        identity,
        triu_indices,
        repeat,
        bincount,
        s_,
        nan_to_num,
        copy,
        take_along_axis,
        atleast_1d,
        clip,
        mod,
    )
    import jax.numpy as jnp
    from jax.numpy import max as module_max
    from jax.numpy import min as module_min
    from jax.numpy.fft import rfft
    from jax.scipy.integrate import trapezoid as trapz
    from jax.scipy.linalg import block_diag

    # loadtxt is not available in jax.numpy; provide a numpy fallback
    from numpy import loadtxt

    # taken from https://github.com/google/jax/issues/2680
    def fill_diagonal(a, val):
        assert a.ndim >= 2
        i, j = jnp.diag_indices(min(a.shape[-2:]))
        return a.at[..., i, j].set(val)

else:
    logger.debug("cmb_baryon_emu.module: Using NumPy backend (numpy)")
    import numpy

    from numpy import (
        isnan,
        isinf,
        savez,
        load,
        ndarray,
        conj,
        ones,
        tan,
        log,
        log10,
        logspace,
        swapaxes,
        empty,
        array,
        linspace,
        arange,
        delete,
        where,
        pi,
        cos,
        sin,
        exp,
        sqrt,
        trapz,
        concatenate,
        linalg,
        eye,
        einsum,
        einsum_path,
        zeros,
        sum,
        pad,
        diag,
        block,
        array_equal,
        meshgrid,
        geomspace,
        moveaxis,
        ones_like,
        empty_like,
        real,
        zeros_like,
        float32,
        float64,
        int32,
        dot,
        multiply,
        add,
        subtract,
        unique,
        hstack,
        isin,
        newaxis,
        ix_,
        transpose,
        interp,
        rollaxis,
        logical_or,
        inf,
        tanh,
        column_stack,
        power,
        asarray,
        loadtxt,
        sign,
        matmul,
        logical_and,
        stack,
        all,
        any,
        mean,
        ndim,
        ceil,
        argwhere,
        cov,
        nansum,
        nanmax,
        fill_diagonal,
        identity,
        triu_indices,
        repeat,
        bincount,
        s_,
        nan_to_num,
        copy,
        take_along_axis,
        atleast_1d,
        clip,
        mod,
    )
    from numpy import max as module_max
    from numpy import min as module_min
    from numpy.fft import rfft
    from scipy.linalg import block_diag
    import numpy as jnp  # alias so JAX-style `jnp.` calls work in numpy mode


# Explicit __all__ makes the wildcard export deliberate and auditable.
# Every symbol listed here is importable from both the JAX and NumPy branches.
__all__ = [
    "config", "is_jax",
    # array creation / type constants
    "array", "zeros", "ones", "empty", "eye", "identity",
    "zeros_like", "ones_like", "empty_like", "asarray",
    "linspace", "arange", "logspace", "geomspace",
    "ndarray", "float32", "float64", "int32",
    "pi", "inf", "newaxis", "s_",
    # math
    "log", "log10", "exp", "sqrt", "sin", "cos", "tan", "tanh",
    "power", "mod", "sign", "real", "conj", "clip",
    "add", "subtract", "multiply", "dot", "matmul",
    "einsum", "einsum_path",
    # reductions
    "sum", "mean", "nansum", "nanmax", "cov", "all", "any",
    "module_max", "module_min",
    # shape / indexing
    "concatenate", "stack", "hstack", "column_stack",
    "swapaxes", "moveaxis", "rollaxis", "transpose",
    "pad", "diag", "block", "block_diag",
    "repeat", "atleast_1d",
    "ix_", "take_along_axis", "triu_indices",
    "meshgrid", "unique", "delete", "interp",
    # logic / comparison
    "isnan", "isinf", "where", "logical_and", "logical_or", "isin",
    "array_equal", "argwhere",
    # linear algebra
    "linalg", "fill_diagonal",
    # I/O
    "savez", "load", "loadtxt",
    # misc
    "bincount", "nan_to_num", "copy",
    "ndim", "ceil", "rfft", "trapz",
    # JAX-compatible alias (numpy in numpy-mode, jax.numpy in JAX-mode)
    "jnp",
]
