from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class _RunningMSE(nn.Module):
    """
    Minimal running MSE metric compatible with the way NequIPLightningModule
    interacts with loss metric objects.
    """

    def __init__(self):
        super().__init__()
        self.dist_sync_on_step = False
        self.register_buffer("sum", torch.tensor(0.0))
        self.register_buffer("count", torch.tensor(0.0))

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        diff2 = (preds - target).pow(2)
        self.sum = self.sum + diff2.sum()
        self.count = self.count + torch.tensor(
            float(diff2.numel()), device=diff2.device, dtype=self.count.dtype
        )

    def compute(self) -> torch.Tensor:
        if self.count.item() == 0:
            return torch.tensor(0.0, device=self.sum.device, dtype=self.sum.dtype)
        return self.sum / self.count

    def reset(self):
        self.sum.zero_()
        self.count.zero_()


class SpinDualLoss(nn.Module):
    """
    NequIP-compatible loss manager for:
      - E_singlet
      - f_singlet
      - E_triplet
      - f_triplet

    This behaves like a small MetricsManager:
      - do_weighted_sum
      - values()
      - __call__(..., prefix=...)
      - compute(prefix=...)
      - reset()
    """

    def __init__(
        self,
        coeffs: Optional[Dict[str, float]] = None,
        per_atom_energy: bool = False,
        type_names=None,
    ):
        super().__init__()
        self.coeffs = coeffs or {
            "E_singlet": 1.0,
            "f_singlet": 1.0,
            "E_triplet": 1.0,
            "f_triplet": 1.0,
        }
        self.per_atom_energy = per_atom_energy
        self.do_weighted_sum = True

        self.metrics = nn.ModuleDict({
            "E_singlet": _RunningMSE(),
            "f_singlet": _RunningMSE(),
            "E_triplet": _RunningMSE(),
            "f_triplet": _RunningMSE(),
        })

    def values(self):
        return self.metrics.values()

    def _maybe_per_atom(self, energy: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.per_atom_energy:
            return energy

        if "ptr" in batch:
            natoms = batch["ptr"][1:] - batch["ptr"][:-1]
            natoms = natoms.to(energy.device).to(energy.dtype)
            return energy / natoms

        if "natoms" in batch:
            natoms = batch["natoms"].to(energy.device).to(energy.dtype)
            return energy / natoms

        raise KeyError("per_atom_energy=True but neither 'ptr' nor 'natoms' found in batch.")

    def _prepare(self, pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        missing_pred = [k for k in self.metrics.keys() if k not in pred]
        missing_ref = [k for k in self.metrics.keys() if k not in batch]
        if missing_pred:
            raise KeyError(f"Missing prediction keys: {missing_pred}")
        if missing_ref:
            raise KeyError(f"Missing batch keys: {missing_ref}")

        prepared = {
            "E_singlet": (
                self._maybe_per_atom(pred["E_singlet"], batch),
                self._maybe_per_atom(batch["E_singlet"], batch),
            ),
            "f_singlet": (pred["f_singlet"], batch["f_singlet"]),
            "E_triplet": (
                self._maybe_per_atom(pred["E_triplet"], batch),
                self._maybe_per_atom(batch["E_triplet"], batch),
            ),
            "f_triplet": (pred["f_triplet"], batch["f_triplet"]),
        }
        return prepared

    def __call__(self, pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], prefix: str = ""):
        prepared = self._prepare(pred, batch)

        out = {}
        weighted_sum = None

        for name, (p, t) in prepared.items():
            self.metrics[name].update(p, t)
            value = ((p - t).pow(2)).mean()
            out[f"{prefix}{name}"] = value

            coeff = self.coeffs.get(name, 0.0)
            weighted_term = coeff * value
            weighted_sum = weighted_term if weighted_sum is None else (weighted_sum + weighted_term)

        out[f"{prefix}weighted_sum"] = weighted_sum
        return out

    def compute(self, prefix: str = ""):
        out = {}
        weighted_sum = None

        for name, metric in self.metrics.items():
            value = metric.compute()
            out[f"{prefix}{name}"] = value

            coeff = self.coeffs.get(name, 0.0)
            weighted_term = coeff * value
            weighted_sum = weighted_term if weighted_sum is None else (weighted_sum + weighted_term)

        out[f"{prefix}weighted_sum"] = weighted_sum
        return out

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()