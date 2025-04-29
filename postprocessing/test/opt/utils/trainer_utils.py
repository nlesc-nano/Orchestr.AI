import os
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from utils.helpers import get_optimizer_class, get_scheduler_class 
import schnetpack as spk

def setup_task_and_trainer(config, nnpot, outputs, folder):
    optimizer_name = config['settings']['training']['optimizer']['type']
    scheduler_name = config['settings']['training']['scheduler']['type']

    optimizer_cls = get_optimizer_class(optimizer_name)
    scheduler_cls = get_scheduler_class(scheduler_name)
    
    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=outputs,
        optimizer_cls=optimizer_cls,
        optimizer_args={"lr": config['settings']['training']['optimizer']['lr']},
        scheduler_cls=scheduler_cls,
        scheduler_args={"mode": "min", "factor": config['settings']['training']['scheduler']['factor'],
                        "patience": config['settings']['training']['scheduler']['patience'],
                        "verbose": config['settings']['training']['scheduler']['verbose']},
        scheduler_monitor=config['settings']['logging']['monitor']
    )
    
    tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=folder)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(folder, config['settings']['logging']['checkpoint_dir']),
            save_top_k=1,
            monitor=config['settings']['logging']['monitor']
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=tensorboard_logger,
        default_root_dir=folder,
        max_epochs=config['settings']['training']['max_epochs'],
        accelerator=config['settings']['training']['accelerator'],
        precision=config['settings']['training']['precision'],
        devices=config['settings']['training']['devices']
    )
    
    logging.info("Task and trainer set up")
    return task, trainer