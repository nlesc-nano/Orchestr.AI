from __future__ import annotations

from typing import Dict, Optional

from nequip.train import NequIPLightningModule


class SpinDualLightningModule(NequIPLightningModule):
    """
    Minimal NequIP-compatible training module for dual singlet/triplet targets.
    """

    def __init__(
        self,
        model: Dict,
        num_datasets: Dict[str, int],
        optimizer: Optional[Dict] = None,
        lr_scheduler: Optional[Dict] = None,
        loss: Optional[Dict] = None,
        train_metrics: Optional[Dict] = None,
        val_metrics: Optional[Dict] = None,
        test_metrics: Optional[Dict] = None,
        info_dict: Optional[Dict] = None,
    ):
        super().__init__(
            model=model,
            num_datasets=num_datasets,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss=loss,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            info_dict=info_dict,
        )
        self._printed_target_keys = False

    def process_target(self, batch, batch_idx: int, dataloader_idx: int = 0):
        target = batch.copy()

        # Print keys once for debugging
        if not self._printed_target_keys:
            print("\n[SpinDual DEBUG] Batch keys from NequIP datamodule:")
            print(sorted(list(target.keys())))
            self._printed_target_keys = True

        # Alias mapping: add more cases if needed after seeing printed keys
        alias_map = {
            "E_singlet": ["E_singlet", "e_singlet", "energy_singlet", "singlet_energy"],
            "E_triplet": ["E_triplet", "e_triplet", "energy_triplet", "triplet_energy"],
            "f_singlet": ["f_singlet", "forces_singlet", "singlet_forces"],
            "f_triplet": ["f_triplet", "forces_triplet", "triplet_forces"],
        }

        normalized = {}
        for wanted_key, candidates in alias_map.items():
            found = None
            for cand in candidates:
                if cand in target:
                    found = cand
                    break
            if found is not None:
                normalized[wanted_key] = target[found]

        # Keep original keys too
        target.update(normalized)
        return target