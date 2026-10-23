"""clvla: infrastructure for studying how pi0.5-LIBERO continually learns LIBERO-90 skills.

The package is deliberately thin. Model code, transforms and the policy server come from the pinned OpenPI checkout
(third_party/openpi); task definitions and the simulator come from the pinned official LIBERO checkout
(third_party/openpi/third_party/libero). Everything here must stay importable on Python 3.8 (simulator env).
"""

__version__ = "0.1.0"
