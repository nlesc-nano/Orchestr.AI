
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import logging
import os

def train_model(config, custom_data, task):
    folder = config['settings']['logging']['folder']
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, "lightning_logs"), exist_ok=True)

    logger = pl.loggers.TensorBoardLogger(save_dir=folder)

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(folder, config['settings']['logging']['checkpoint_dir']),
            save_top_k=1,
            monitor=config['settings']['logging']['monitor']
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=folder,
        max_epochs=config['settings']['training']['max_epochs'],
        accelerator=config['settings']['training']['accelerator'],
        precision=config['settings']['training']['precision'],
        devices=config['settings']['training']['devices']
    )

    logging.info("Start training")
    trainer.fit(task, datamodule=custom_data)
