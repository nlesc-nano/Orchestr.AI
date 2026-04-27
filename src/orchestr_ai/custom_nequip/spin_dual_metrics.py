from __future__ import annotations

import torch
import torch.nn as nn


class _RunningMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.dist_sync_on_step = False
        self.register_buffer("sum_abs", torch.tensor(0.0))
        self.register_buffer("count", torch.tensor(0.0))

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        diff = torch.abs(preds - target)
        self.sum_abs = self.sum_abs + diff.sum()
        self.count = self.count + torch.tensor(
            float(diff.numel()), device=diff.device, dtype=self.count.dtype
        )

    def compute(self) -> torch.Tensor:
        if self.count.item() == 0:
            return torch.tensor(0.0, device=self.sum_abs.device, dtype=self.sum_abs.dtype)
        return self.sum_abs / self.count

    def reset(self):
        self.sum_abs.zero_()
        self.count.zero_()


class SpinDualMetrics(nn.Module):
    """
    NequIP-compatible metric manager for:
      - E_singlet
      - f_singlet
      - E_triplet
      - f_triplet
    """

    def __init__(self, type_names=None):
        super().__init__()
        self.metrics = nn.ModuleDict({
            "E_singlet_mae": _RunningMAE(),
            "f_singlet_mae": _RunningMAE(),
            "E_triplet_mae": _RunningMAE(),
            "f_triplet_mae": _RunningMAE(),
        })

    def values(self):
        return self.metrics.values()

    def __call__(self, pred, batch, prefix: str = ""):
        required = {
            "E_singlet_mae": ("E_singlet", "E_singlet"),
            "f_singlet_mae": ("f_singlet", "f_singlet"),
            "E_triplet_mae": ("E_triplet", "E_triplet"),
            "f_triplet_mae": ("f_triplet", "f_triplet"),
        }

        out = {}
        weighted_sum = None

        for metric_name, (pred_key, ref_key) in required.items():
            if pred_key not in pred:
                raise KeyError(f"Missing prediction key: {pred_key}")
            if ref_key not in batch:
                raise KeyError(f"Missing batch key: {ref_key}")

            self.metrics[metric_name].update(pred[pred_key], batch[ref_key])
            value = torch.mean(torch.abs(pred[pred_key] - batch[ref_key]))
            out[f"{prefix}{metric_name}"] = value
            weighted_sum = value if weighted_sum is None else (weighted_sum + value)

        out[f"{prefix}weighted_sum"] = weighted_sum
        return out

    def compute(self, prefix: str = ""):
        out = {}
        weighted_sum = None

        for name, metric in self.metrics.items():
            value = metric.compute()
            out[f"{prefix}{name}"] = value
            weighted_sum = value if weighted_sum is None else (weighted_sum + value)

        out[f"{prefix}weighted_sum"] = weighted_sum
        return out

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()