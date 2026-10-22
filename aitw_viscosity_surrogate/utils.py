"""Utility functions for the viscosity surrogate model."""
import sys

import numpy as np

from . import loggers as lg

logger = lg.MAIN_LOGGER

_INSTALLED = False

def numpy_1x_compatibility():
    """The default models are saved with numpy 2.x. Attempting to unpickle them with numpy 1.x will result in a error
    due to the change in some module definitions."""
    global _INSTALLED
    if _INSTALLED:
        return

    _INSTALLED = True

    if np.__version__ < '2.':
        logger.warning(
            f"NumPy version {np.__version__} detected. "
            f"Consider upgrading to NumPy 2.x for better performance and compatibility. "
            "Attempting to use compatibility wrappers"
        )
        sys.modules['numpy._core'] = sys.modules['numpy.core']
        for name in ("multiarray", "numeric", " umath"):
            old_name = f"numpy.core.{name.strip()}"
            new_name = f"numpy._core.{name.strip()}"
            try:
                sys.modules[new_name] = __import__(old_name, fromlist=["*"])
            except ImportError:
                pass
