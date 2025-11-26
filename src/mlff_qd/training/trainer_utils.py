import os
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from mlff_qd.utils.callbacks import StopWhenLRBelow, StopWhenGoodEnough, EarlyStoppingWithLog
import schnetpack as spk
from mlff_qd.utils.yaml_utils import _int_or_len_devices 

from mlff_qd.utils.helpers import get_optimizer_class, get_scheduler_class 

def setup_task_and_trainer(config, nnpot, outputs, folder):
    optimizer_name = config['training']['optimizer']['type']
    scheduler_name = config['training']['scheduler']['type']

    optimizer_cls = get_optimizer_class(optimizer_name)
    scheduler_cls = get_scheduler_class(scheduler_name)
    
    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=outputs,
        optimizer_cls=optimizer_cls,
        optimizer_args={"lr": config['training']['optimizer']['lr']},
        scheduler_cls=scheduler_cls,
        scheduler_args={"mode": "min", "factor": config['training']['scheduler']['factor'],
                        "patience": config['training']['scheduler']['patience'],
                        "verbose": config['training']['scheduler']['verbose']},
        scheduler_monitor=config['logging']['monitor']
    )
    
    tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=folder)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(folder, config['logging']['checkpoint_dir']),
            save_top_k=1,
            monitor=config['logging']['monitor']
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]

    # EarlyStopping ---------------------------------------------------------
    if 'early_stopping' in config['training']:
        es = config['training']['early_stopping']
        callbacks.append(
            EarlyStoppingWithLog(
                monitor = es['monitor'],
                patience = es['patience'],
                min_delta = es.get('min_delta', 0.),
                mode     = es.get('mode', 'min'),
                verbose  = True
            )
        )

    # LR threshold ----------------------------------------------------------
    lr_thr = config['training'].get('lr_stop_threshold')
    if lr_thr is not None:
        callbacks.append(StopWhenLRBelow(threshold=lr_thr))

    # Accuracy target -------------------------------------------------------
    tgt_cfg = config['training'].get('targets', {}).get('stop_when_good_enough')
    if tgt_cfg:
        callbacks.append(
            StopWhenGoodEnough(
                monitor = tgt_cfg['monitor'],
                target  = tgt_cfg['threshold']
            )
        )

    print("CALLBACKS:", callbacks)
    
    
    accelerator = config['training'].get('accelerator', 'auto')
    devices_raw = config['training'].get('devices', 1)
    devs = _int_or_len_devices(devices_raw)

    strategy = None
    num_nodes = None

    if devs is None:
        # devices is "auto" or unknown → let Lightning decide silently
        pass
    elif devs >= 2:
        strategy = "ddp"
        num_nodes = config['training'].get('num_nodes', 1)
        logging.info("[Trainer] Enabling DDP: devices=%s, strategy=ddp, num_nodes=%s", devices_raw, num_nodes)

    trainer_kwargs = dict(
        callbacks=callbacks,
        logger=tensorboard_logger,
        default_root_dir=folder,
        max_epochs=config['training']['max_epochs'],
        accelerator=accelerator,
        precision=config['training']['precision'],
        devices=devices_raw,              # int, list, or "auto" all OK
        num_sanity_val_steps=0,
        log_every_n_steps=config['training']['log_every_n_steps'],
        enable_progress_bar=True,
    )
    if strategy is not None:
        trainer_kwargs["strategy"] = strategy
    if num_nodes is not None:
        trainer_kwargs["num_nodes"] = num_nodes
    
    trainer = pl.Trainer(**trainer_kwargs)
    logging.info("Task and trainer set up")
    
    return task, trainer
