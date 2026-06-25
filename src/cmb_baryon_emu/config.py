"""Configuration module for managing JAX/NumPy backend switching.

This module provides a simple interface to enable or disable JAX backend
usage across the UPanda package. When JAX is enabled, operations use
JAX arrays and functions for improved performance with JIT compilation
and automatic differentiation.

Attributes:
    is_jax_enabled (bool): Global flag indicating whether JAX backend
        is currently enabled. Defaults to False.
"""

import importlib
import logging
import sys

logger = logging.getLogger(__name__)

is_jax_enabled = False  # Default setting


def set_jax_enabled(value: bool):
    """Enable or disable JAX backend globally.

    This function sets the JAX backend flag and automatically reloads
    all UPanda modules that depend on the backend selection to ensure
    they pick up the correct numpy/jax functions.

    Args:
        value (bool): True to enable JAX backend, False to use NumPy.
    """
    global is_jax_enabled
    old_value = is_jax_enabled
    is_jax_enabled = value

    # If the value changed, reload modules to pick up the new backend
    if old_value != value:
        _reload_modules()


def get_jax_enabled():
    """Get the current JAX backend status.

    Returns:
        bool: True if JAX is enabled, False if using NumPy.
    """
    return is_jax_enabled


def _reload_modules():
    """Reload UPanda modules after JAX backend change.

    This ensures that all modules pick up the correct numpy/jax functions
    after the backend is switched. Modules are reloaded in dependency order.
    """
    # List of modules to reload in order (dependencies first)
    modules_to_reload = [
        "cmb_baryon_emu.module",
        "cmb_baryon_emu.integrated_model_jax",
        # Add other modules here if needed
    ]

    for module_name in modules_to_reload:
        if module_name in sys.modules:
            try:
                importlib.reload(sys.modules[module_name])
                logger.debug(
                    "Reloaded %s with JAX=%s",
                    module_name,
                    "enabled" if is_jax_enabled else "disabled",
                )
            except Exception as e:
                logger.warning("Could not reload %s: %s", module_name, e)
