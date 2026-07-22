"""ALIS: Absorption LIne Software.

This module holds the single source of truth for the package version.
``pyproject.toml`` derives its version from ``__version__`` here via
setuptools' dynamic ``attr`` mechanism, so the string need only be
updated in one place.
"""

__version__ = "2.0.0.dev0"
