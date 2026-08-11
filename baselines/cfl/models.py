"""Model definition for the CFL baseline (matches the reference implementation)."""

from __future__ import annotations

import torch
import torch.nn as nn


class CFLNet(nn.Module):
    """The reference's simple MLP: 784 -> 100 -> 100 -> 10 with tanh."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        return self.net(x)


def load_parameters_from_state_dict(model: torch.nn.Module) -> list:
    """Extract a model's parameters as a list of NumPy arrays (ordered)."""
    return [p.detach().cpu().numpy() for p in model.state_dict().values()]


def load_state_dict_from_parameters(model: torch.nn.Module, ndarrays: list) -> None:
    """Load a list of NumPy arrays into ``model`` by state_dict key order."""
    import torch  # noqa: F811

    names = list(model.state_dict().keys())
    state_dict = {
        name: torch.from_numpy(arr) for name, arr in zip(names, ndarrays)
    }
    model.load_state_dict(state_dict)
