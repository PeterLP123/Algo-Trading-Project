"""Backward-compatible facade for the refactored research package.

Existing notebooks and scripts may continue importing from strategy_helpers.
New code should import from the focused modules under systematic_crypto.
"""

from systematic_crypto import *  # noqa: F401,F403
from systematic_crypto import __all__
