"""pytest configuration for the mimosa-fl test suite.

Ensures the repository root is importable regardless of how pytest was
invoked, and declares reusable fixtures.
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture
def model_parameters():
    """Small model parameters used across registry/strategy tests."""
    from tests.mimosa.fixtures.parameters import small_parameters

    return small_parameters()
