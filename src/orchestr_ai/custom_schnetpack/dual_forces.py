from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import grad
import schnetpack.properties as properties


class DualForces(nn.Module):
    """
    Compute two force fields from two energy outputs in one pass.
    """

    def __init__(
        self,
        energy_key_1: str = "E_singlet",
        force_key_1: str = "f_singlet",
        energy_key_2: str = "E_triplet",
        force_key_2: str = "f_triplet",
    ):
        super().__init__()
        self.energy_key_1 = energy_key_1
        self.force_key_1 = force_key_1
        self.energy_key_2 = energy_key_2
        self.force_key_2 = force_key_2

        self.required_derivatives = [properties.R]
        self.model_outputs = [self.force_key_1, self.force_key_2]

    def forward(self, inputs):
        if self.energy_key_1 not in inputs:
            raise KeyError(f"Missing energy key: {self.energy_key_1}")
        if self.energy_key_2 not in inputs:
            raise KeyError(f"Missing energy key: {self.energy_key_2}")
        if properties.R not in inputs:
            raise KeyError(f"Missing position key: {properties.R}")

        positions = inputs[properties.R]
        e1 = inputs[self.energy_key_1]
        e2 = inputs[self.energy_key_2]

        # First derivative must ALWAYS keep graph alive for second derivative
        f1 = -grad(
            e1.sum(),
            positions,
            create_graph=self.training,
            retain_graph=True,
        )[0]

        # Second derivative is the last one in forward
        # keep graph only if training, because backward still needs it
        f2 = -grad(
            e2.sum(),
            positions,
            create_graph=self.training,
            retain_graph=self.training,
        )[0]

        inputs[self.force_key_1] = f1
        inputs[self.force_key_2] = f2
        return inputs