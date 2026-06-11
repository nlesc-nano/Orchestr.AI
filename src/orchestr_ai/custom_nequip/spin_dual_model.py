from __future__ import annotations

from typing import Dict, Any, List

import torch
import torch.nn as nn


class SpinDualHeadNequIPModel(nn.Module):
    """
    Minimal NequIP-compatible dual-head graph model prototype.

    This is still a prototype, but now it exposes the attributes that
    NequIPLightningModule expects:
      - is_graph_model
      - type_names

    It predicts:
      - E_singlet
      - E_triplet
      - f_singlet
      - f_triplet

    IMPORTANT:
    This is not yet a true NequIP equivariant model.
    It is only the next interface-compatible step.
    """

    is_graph_model = True

    def __init__(
        self,
        type_names: List[str],
        num_features: int = 32,
        **kwargs: Any,
    ):
        super().__init__()

        self.type_names = list(type_names)
        hidden_dim = int(num_features)

        # simple prototype backbone over positions
        self.node_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.energy_shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.singlet_head = nn.Linear(hidden_dim, 1)
        self.triplet_head = nn.Linear(hidden_dim, 1)

    def _graph_sum(self, node_values: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Sum per-atom values into per-graph values.
        Expects 'batch' if multiple graphs are present.
        """
        if "batch" not in data:
            return node_values.sum(dim=0, keepdim=True)

        batch_index = data["batch"]
        n_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 1

        out = torch.zeros(
            n_graphs,
            node_values.shape[-1],
            device=node_values.device,
            dtype=node_values.dtype,
        )
        out.index_add_(0, batch_index, node_values)
        return out

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # NequIP data commonly uses "pos"
        if "pos" not in data:
            raise KeyError("SpinDualHeadNequIPModel expects data['pos'].")

        pos = data["pos"]
        if not pos.requires_grad:
            pos = pos.clone().detach().requires_grad_(True)

        atom_feat = self.node_mlp(pos)
        shared = self.energy_shared(atom_feat)

        atom_e_singlet = self.singlet_head(shared)   # [n_atoms, 1]
        atom_e_triplet = self.triplet_head(shared)   # [n_atoms, 1]

        E_singlet = self._graph_sum(atom_e_singlet, data).squeeze(-1)
        E_triplet = self._graph_sum(atom_e_triplet, data).squeeze(-1)

        f_singlet = -torch.autograd.grad(
            E_singlet.sum(),
            pos,
            create_graph=self.training,
            retain_graph=True,
        )[0]

        f_triplet = -torch.autograd.grad(
            E_triplet.sum(),
            pos,
            create_graph=self.training,
            retain_graph=True,
        )[0]

        return {
            "E_singlet": E_singlet,
            "E_triplet": E_triplet,
            "f_singlet": f_singlet,
            "f_triplet": f_triplet,
        }