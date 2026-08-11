"""Version compatibility shims for the Flower 1.28 - 1.33 API drift.

Flower's modern strategy API (``flwr.serverapp.strategy.Strategy``) is
message-based: it exchanges ``ArrayRecord`` / ``ConfigRecord`` / ``MetricRecord``
objects inside ``Message`` payloads over a ``Grid`` handle. The import locations
of these types moved between releases:

- 1.28 - 1.30: ``flwr.common`` / ``flwr.server``
- 1.31 - 1.33: ``flwr.app`` / ``flwr.serverapp``

This module re-exports them from whichever location is present so that the rest
of mimosa-fl can target the whole supported range.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from flwr.app import (  # type: ignore[attr-defined]
        ArrayRecord,
        ConfigRecord,
        Context,
        Message,
        MessageType,
        MetricRecord,
        RecordDict,
    )
except ImportError:  # pragma: no cover - depends on installed Flower version
    from flwr.common import (  # type: ignore[no-redef]
        ArrayRecord,
        ConfigRecord,
        Context,
        Message,
        MessageType,
        MetricRecord,
        RecordDict,
    )

try:
    from flwr.serverapp import Grid  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    from flwr.server import Grid  # type: ignore[no-redef]

try:
    from flwr.serverapp.strategy import FedAvg, Strategy
except ImportError:  # pragma: no cover - depends on installed Flower version
    FedAvg = None  # type: ignore[assignment]
    Strategy = None  # type: ignore[assignment]

try:
    from flwr.common import (
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
except ImportError:  # pragma: no cover
    ndarrays_to_parameters = None  # type: ignore[assignment]
    parameters_to_ndarrays = None  # type: ignore[assignment]


def to_ndarrays(parameters: Any) -> list[np.ndarray]:
    """Normalize Flower ``Parameters`` (or a plain list) to ``list[ndarray]``."""
    if hasattr(parameters, "tensors"):
        if parameters_to_ndarrays is not None:
            return parameters_to_ndarrays(parameters)
        raise TypeError(
            "Flower `Parameters` received but `parameters_to_ndarrays` is unavailable."
        )
    return [np.asarray(p, copy=True) for p in parameters]


def to_parameters(ndarrays: list[np.ndarray]) -> Any:
    """Normalize ``list[ndarray]`` into Flower ``Parameters`` when available."""
    if ndarrays_to_parameters is not None:
        return ndarrays_to_parameters([np.asarray(a, copy=True) for a in ndarrays])
    return [np.asarray(a, copy=True) for a in ndarrays]


def arrayrecord_to_ndarrays(record: Any) -> list[np.ndarray]:
    """Convert an ``ArrayRecord`` into a plain list of NumPy arrays."""
    if hasattr(record, "to_numpy_ndarrays"):
        return record.to_numpy_ndarrays()
    if hasattr(record, "to_numpy"):
        return record.to_numpy()
    return [v.numpy() for v in record.values()]


def ndarrays_to_arrayrecord(ndarrays: list[np.ndarray]) -> Any:
    """Convert a plain list of NumPy arrays into an ``ArrayRecord``."""
    clean = [np.asarray(a, copy=True) for a in ndarrays]
    try:
        return ArrayRecord(numpy_ndarrays=clean)  # type: ignore[attr-defined]
    except TypeError:  # pragma: no cover - older ArrayRecord constructor
        return ArrayRecord(clean)  # type: ignore[attr-defined]
