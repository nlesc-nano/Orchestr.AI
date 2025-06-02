import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# helper ----------------------------------------------------------------------
def _tb_log(trainer, key, value, global_step=None):
    """Log to TensorBoard if the trainer has a TensorBoard logger."""
    if trainer.logger is None:
        return
    if hasattr(trainer.logger, "experiment"):
        tb = trainer.logger.experiment
        if isinstance(value, (int, float)):
            tb.add_scalar(key, value, global_step or trainer.global_step)
        else:  # text
            tb.add_text(key, str(value), global_step or trainer.global_step)

# ② LR threshold --------------------------------------------------------------
class StopWhenLRBelow(pl.Callback):
    def __init__(self, threshold):
        self.threshold = threshold

    def on_train_epoch_end(self, trainer, pl_module):
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        if lr < self.threshold:
            trainer.should_stop = True
            reason = f"LR {lr:.2e} < {self.threshold:.2e}"
            _tb_log(trainer, "early_stop_code", 2)
            _tb_log(trainer, "early_stop_reason", reason)
            print(f"[StopWhenLRBelow] {reason}", flush=True)

# ③ Accuracy target -----------------------------------------------------------
class StopWhenGoodEnough(pl.Callback):
    def __init__(self, monitor, target):
        self.monitor, self.target = monitor, target

    def on_validation_end(self, trainer, pl_module):
        val = trainer.callback_metrics.get(self.monitor)
        if val is not None and val < self.target:
            trainer.should_stop = True
            reason = f"{self.monitor} {val:.4g} < {self.target}"
            _tb_log(trainer, "early_stop_code", 3)
            _tb_log(trainer, "early_stop_reason", reason)
            print(f"[StopWhenGoodEnough] {reason}", flush=True)

# ① Early stopping with log ---------------------------------------------------
class EarlyStoppingWithLog(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)

        # print/log only on the epoch when stopping is triggered
        if self.stopped_epoch == trainer.current_epoch:
            reason = (
                f"Early stopping: {self.monitor} did not improve for "
                f"{self.patience} epochs (best {self.best_score:.4g})"
            )
            _tb_log(trainer, "early_stop_code", 1)
            _tb_log(trainer, "early_stop_reason", reason)
            print(f"[EarlyStoppingWithLog] {reason}", flush=True)

    # Final safety log in case you miss it during validation
    def on_train_end(self, trainer, pl_module):
        if self.stopped_epoch > 0:
            reason = (
                f"Early stopping: {self.monitor} did not improve for "
                f"{self.patience} epochs (best {self.best_score:.4g})"
            )
            print(f"[EarlyStoppingWithLog] FINAL: {reason}", flush=True)


